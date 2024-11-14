clc;
clear;

% 模擬參數
dt = 0.1; % 時間步長
T = 100; % 總模擬時間
steps = T / dt; % 模擬步數

% Agent 數量
num_agents = 3; 
r_safety = 1; % 安全半徑
r_safety_obstacle = 1;

% 定義障礙物數量、位置和半徑
num_obstacles = 3;  % 定義障礙物的數量
obstacle_positions = [5, 0, 3;   % 障礙物 1 的位置 (X, Y, Z)
                      5, 0, 3;   % 障礙物 2 的位置 (X, Y, Z)
                      5, 0, 3]; % 障礙物 3 的位置 (X, Y, Z)
obstacle_radii = [0.5; 1.5; 0.5];  % 障礙物的半徑

% 鄰接矩陣表示 Agent 之間的通訊，1 表示通訊，0 表示沒有通訊
adjacency_matrix = [0 1 1;
                    1 0 1;
                    1 1 0];

% 虛擬Leader初始位置與目標位置
virtual_leader_initial = [10, 0, 3, 0];  % [x, y, z, theta]
virtual_leader_target = [10, 0, 3, 0]; % [x, y, z, theta]

% 定義基於虛擬Leader的隊形向量
desired_relative_positions = [
    0, 0, 0;     % Agent 1
    -2, 2, 0;    % Agent 2
    -2, -2, 0    % Agent 3
];

% 初始條件矩陣 [x0, y0, z0, theta0]，每一行對應一個 Agent
initial_conditions = [
    0, 0, 3, 0;    % Agent 1 初始位置 (X, Y, Z, theta)
    -2, 4, 3, 0;  % Agent 2 初始位置 (X, Y, Z, theta)
    -2, -4, 3, 0 % Agent 3 初始位置 (X, Y, Z, theta)
];

% MPC 參數
Np = 40; % 預測範圍（時間步）
Nu = 40;
v_max = 5.0; % 最大線速度
omega_max = pi / 8; % 最大角速度

% 狀態變數初始化
x = zeros(steps, num_agents);
y = zeros(steps, num_agents);
z = zeros(steps, num_agents); % Z 軸高度
theta = zeros(steps, num_agents);
v = zeros(steps, num_agents);
omega = zeros(steps, num_agents);
phi = zeros(steps, num_agents); % 修正角度

% 初始狀態設置
x(1, :) = initial_conditions(:, 1);
y(1, :) = initial_conditions(:, 2);
z(1, :) = initial_conditions(:, 3); % 初始 Z 軸高度
theta(1, :) = initial_conditions(:, 4);

% fmincon 的選項
options = optimoptions('fmincon', ...
            'Display', 'off', ...               % 顯示每次迭代的過程（你可以選擇 'off', 'iter', 'final'）
            'MaxIterations', 1000, ...          % 設置最大迭代次數 
            'MaxFunctionEvaluations', 1000, ... % 設置最大函數評估次數
            'OptimalityTolerance', 1e-8, ...    % 調整最優解公差（收斂準確度）
            'StepTolerance', 1e-8, ...          % 調整步長公差（控制最小步長）
            'ConstraintTolerance', 1e-8, ...    % 調整非線性約束的容忍度
            'Algorithm', 'interior-point', ...  % 選擇適合的優化算法（'interior-point', 'sqp','active-set')
            'HessianApproximation', 'lbfgs', ...% 使用Hessian近似，避免奇異問題
            'UseParallel', false);              % 是否啟用並行計算（如果有多核 CPU）

% 圖形設置
figure;
hold on;

% 繪製障礙物
for i = 1:num_obstacles
    [X_sphere, Y_sphere, Z_sphere] = sphere(20);
    X_sphere = X_sphere * obstacle_radii(i) + obstacle_positions(i, 1);
    Y_sphere = Y_sphere * obstacle_radii(i) + obstacle_positions(i, 2);
    Z_sphere = Z_sphere * obstacle_radii(i) + obstacle_positions(i, 3);
    surf(X_sphere, Y_sphere, Z_sphere, 'FaceColor', 'k', 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', ['障礙物 ', num2str(i)]);       
end

% 初始化虛擬Leader的狀態
virtual_leader_position = virtual_leader_initial(1:3); % X, Y, Z
virtual_leader_theta = virtual_leader_initial(4);

% 繪製虛擬Leader的黑色虛線軌跡
hVirtualLeaderLine = animatedline('Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', '虛擬Leader 路徑');


% 繪製Agent軌跡
hTrajectories = gobjects(1, num_agents);
hAgents = gobjects(1, num_agents);       % 小球物件
hArrows = gobjects(1, num_agents);
hPredicted = gobjects(1, num_agents);
hSafetyCircles = gobjects(1, num_agents);
colors = ['b', 'g', 'm'];

for i = 1:num_agents
    hTrajectories(i) = plot3(x(1, i), y(1, i), z(1, i), ['-', colors(i)], 'DisplayName', ['Agent ', num2str(i), ' 移動軌跡']);
    hAgents(i) = scatter3(x(1, i), y(1, i), z(1, i), 10, colors(i), 'filled', 'DisplayName', ['Agent ', num2str(i), ' 本體']);
    hPredicted(i) = plot3(nan(Np, 1), nan(Np, 1), nan(Np, 1), ['--', colors(i)], 'DisplayName', ['Agent ', num2str(i), ' 預測軌跡']); 
    hArrows(i) = quiver3(x(1, i), y(1, i), z(1, i), cos(theta(1, i)), sin(theta(1, i)), 0, 'Color', colors(i), 'MaxHeadSize', 2, 'DisplayName', ['Agent ', num2str(i), ' 偏航角']);                         
    theta_circle = linspace(0, 2 * pi, 100); % 100 points to draw a circle
    x_circle = r_safety * cos(theta_circle);
    y_circle = r_safety * sin(theta_circle);
    hSafetyCircles(i) = plot3(x_circle + x(1, i), y_circle + y(1, i), z(1, i) * ones(size(x_circle)), ...
                              'Color', colors(i), 'LineStyle', '--', 'LineWidth', 0.1, ...
                              'DisplayName', ['Agent ', num2str(i), ' 安全半徑']);
end

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('編隊');
grid on;
axis([-5 15 -15 15 -5 15]);
view(0, 90);
rotate3d on;
video_filename = 'Formation.mp4';
video_writer = VideoWriter(video_filename, 'MPEG-4');
video_writer.FrameRate = 10;
open(video_writer);
stop_animation = false;  % 初始化停止動畫的變數

% 定義虛擬Leader的速度大小
v_leader = 2.0; % 虛擬Leader的直線速度 (可調整)

 
% 初始化一個布林變數以追蹤虛擬Leader是否已到達目標
virtual_leader_reached_target = false;

% 模擬迴圈
for k = 1:steps-1
    % 計算虛擬Leader朝向目標位置的方向
    direction_vector = virtual_leader_target(1:2) - virtual_leader_position(1:2);
    direction_norm = norm(direction_vector); % 計算方向向量的大小
    
    % 檢查虛擬Leader是否已到達目標位置
    if direction_norm < 0.1
        % 若虛擬Leader到達目標位置，設定朝向角為最終目標方向，並停止位置更新
        if ~virtual_leader_reached_target
            virtual_leader_reached_target = true;  % 設為已到達
            disp('虛擬Leader已到達目標位置');  % 顯示訊息一次
            % 設定最終的朝向角為虛擬Leader的移動方向（目標方向）
            virtual_leader_theta = atan2(virtual_leader_target(2) - virtual_leader_initial(2), ...
                                        virtual_leader_target(1) - virtual_leader_initial(1));
        end
        % 停止更新虛擬Leader位置
        virtual_leader_position(1:2) = virtual_leader_target(1:2);
    else
        % 若尚未到達目標位置，繼續沿著目標位置的方向移動
        direction_unit = direction_vector / direction_norm; % 單位方向向量
        virtual_leader_position(1:2) = virtual_leader_position(1:2) + direction_unit * v_leader * dt;
        % 更新虛擬Leader的偏航角
        virtual_leader_theta = atan2(virtual_leader_target(2) - virtual_leader_position(2), virtual_leader_target(1) - virtual_leader_position(1));
    end
    % 更新虛擬Leader黑色虛線軌跡
    addpoints(hVirtualLeaderLine, virtual_leader_position(1), virtual_leader_position(2), virtual_leader_position(3));

    % 更新Agent的目標位置 (跟隨虛擬Leader的隊形)
    for i = 1:num_agents
        % 計算旋轉後的隊形向量
        rotation_matrix = [cos(virtual_leader_theta), -sin(virtual_leader_theta); 
                           sin(virtual_leader_theta), cos(virtual_leader_theta)];
        relative_position = rotation_matrix * desired_relative_positions(i, 1:2)'; % 隊形旋轉後的位置
        % 加入虛擬Leader的位置，得到Agent的目標位置
        agent_target = virtual_leader_position(1:2) + relative_position';
        
        % 設定目標函數 
        cost_function = @(u) mpc_cost_function(x(k, i), y(k, i), theta(k, i), u, ...
                                                agent_target(1), agent_target(2), ...
                                                virtual_leader_theta, Np, Nu, dt);

        % 設定控制輸入的初值與邊界
        u0 = [0; 0; 0]; 
        lb = [-v_max; -omega_max; -pi]; 
        ub = [v_max; omega_max; pi]; 
        
        % 使用 fmincon 計算最佳控制輸入
         nonlincon = @(u) collision_constraint(u, x(k, :), y(k, :), theta(k, :), i, Np, dt, num_agents, ...
                                          adjacency_matrix, r_safety, obstacle_positions, obstacle_radii, r_safety_obstacle);

        u_opt = fmincon(cost_function, u0, [], [], [], [], lb, ub, nonlincon, options);
    
        % 儲存最佳控制輸入
        v(k, i) = u_opt(1);
        omega(k, i) = u_opt(2);
        phi(k, i) = u_opt(3);
        
        % 更新每個 Agent 的狀態 
        x(k+1, i) = x(k, i) + v(k, i) * cos(theta(k, i) + phi(k, i)) * dt;
        y(k+1, i) = y(k, i) + v(k, i) * sin(theta(k, i) + phi(k, i)) * dt;
        z(k+1, i) = z(k, i); 
        theta(k+1, i) = theta(k, i) + omega(k, i) * dt;
        
        % 計算每個Agent的Np步預測軌跡
        x_pred = zeros(Np, 1);
        y_pred = zeros(Np, 1);
        z_pred = z(k, i) * ones(Np, 1);
        theta_pred = theta(k, i);
        x_pred(1) = x(k+1, i);
        y_pred(1) = y(k+1, i);
        
        for j = 2:Np
            x_pred(j) = x_pred(j-1) + v(k, i) * cos(theta_pred + phi(k, i)) * dt;
            y_pred(j) = y_pred(j-1) + v(k, i) * sin(theta_pred + phi(k, i)) * dt;
            theta_pred = theta_pred + omega(k, i) * dt;
        end

        % 更新軌跡和箭頭
        set(hSafetyCircles(i), 'XData', x_circle + x(k+1, i), 'YData', y_circle + y(k+1, i), 'ZData', z(k+1, i) * ones(size(x_circle)));
        set(hTrajectories(i), 'XData', x(1:k+1, i), 'YData', y(1:k+1, i), 'ZData', z(1:k+1, i));
        set(hAgents(i), 'XData', x(k+1, i), 'YData', y(k+1, i), 'ZData', z(k+1, i));
        set(hPredicted(i), 'XData', x_pred, 'YData', y_pred, 'ZData', z_pred); 
        set(hArrows(i), 'XData', x(k+1, i), 'YData', y(k+1, i), 'ZData', z(k+1, i), ...
                        'UData', cos(theta(k+1, i)), 'VData', sin(theta(k+1, i)), 'WData', 0);
    end



    % 捕捉幀並更新圖像
    frame = getframe(gcf);
    writeVideo(video_writer, frame);
    drawnow;
    % 檢查是否有按鍵被按下
    if ~isempty(get(gcf,'CurrentCharacter'))  % 如果有按鍵被按下
        disp('手動終止動畫');
        stop_animation = true;
    end
    
    % 判斷是否需要停止動畫
    if stop_animation
        break;
    end
    pause(0.01)
% 顯示最終軌跡
legend('Location', 'eastoutside');
end
% 關閉並保存影片
close(video_writer);

disp(['影片保存為: ', video_filename]);


% 修改 mpc_cost_function，增加 phi 參數 (控制輸入計算中使用 phi)
function cost = mpc_cost_function(x, y, theta, u, x_target, y_target, theta_target, Np, Nu, dt)
    cost = 0;
    
    % 狀態和控制與輸入權重矩陣
    Q = 20; %導航誤差的權重 距離&航向
    R = 1; %控制輸入的權重
    Qf = 10; %終端誤差的權重
    R_eff = 0.1; %控制變化的權重 速度&角速度&修正角

    u_prev = u;

    for j = 1:Np-1
        if j <= Nu
            v = u(1);
            omega = u(2);
            phi = u(3);
        else
            v = u_prev(1); 
            omega = u_prev(2);
            phi = u_prev(3);
        end

        x = x + v * cos(theta + phi) * dt;
        y = y + v * sin(theta + phi) * dt;
        theta = theta + omega * dt;
        
        % 計算導航誤差和控制輸入的成本
        distance_to_target = sqrt((x - x_target)^2 + (y - y_target)^2);
        heading_error = abs(theta - theta_target); 
        control_effort = v^2 + omega^2 + phi^2;
        
        % 計算控制變化量
        delta_u = [v; omega; phi] - u_prev;
        control_change_cost = delta_u(1)^2 + delta_u(2)^2 + delta_u(3)^2;

        % 累加導航誤差、控制輸入、偏航角誤差和控制變化的成本
        cost = cost + (distance_to_target^2 * Q + heading_error^2 * Q + control_effort * R + control_change_cost * R_eff);

        u_prev = [v; omega; phi];
    end
    
    terminal_distance_to_target = sqrt((x - x_target)^2 + (y - y_target)^2);
    terminal_heading_error = abs(theta - theta_target); 
    cost = cost + (terminal_distance_to_target^2 * Qf + terminal_heading_error^2 * Qf);
end

function [c, ceq] = collision_constraint(u, x, y, theta, agent_index, Np, dt, num_agents, adjacency_matrix, r_safety, obstacle_positions, obstacle_radii, r_safety_obstacle)
    % 計算最大約束數量（代理碰撞 + 障礙物碰撞）
    max_constraints = Np * (num_agents - 1) * num_agents / 2 + Np * length(obstacle_radii); 
    c = zeros(max_constraints, 1);
    ceq = []; 
    constraint_index = 1; 

    % 初始化多代理的預測位置
    x_pred = zeros(Np, num_agents);
    y_pred = zeros(Np, num_agents);
    theta_pred = zeros(Np, num_agents);
    
    % 將所有代理的初始位置設置為當前位置
    x_pred(1, :) = x;
    y_pred(1, :) = y;
    theta_pred(1, :) = theta;

    % 預測所有代理的軌跡
    for n = 2:Np
        for i = 1:num_agents
            v = u(1); 
            omega = u(2); 
            phi = u(3);
            x_pred(n, i) = x_pred(n-1, i) + v * cos(theta_pred(n-1, i) + phi) * dt;
            y_pred(n, i) = y_pred(n-1, i) + v * sin(theta_pred(n-1, i) + phi) * dt;
            theta_pred(n, i) = theta_pred(n-1, i) + omega * dt;
        end
    end

    % 檢查指定代理與其他代理的距離約束
    for j = 1:num_agents
        if adjacency_matrix(agent_index, j) == 1 && agent_index ~= j
            for n = 1:Np
                for m = 1:Np
                    % 計算 agent_index 在第 n 步與 j 在第 m 步之間的距離
                    distance_agents = sqrt((x_pred(n, agent_index) - x_pred(m, j))^2 + (y_pred(n, agent_index) - y_pred(m, j))^2);
                
                    % 碰撞約束：對距離加入安全半徑
                    c(constraint_index) = 2 * r_safety - distance_agents; 
                    constraint_index = constraint_index + 1;
                end
            end
        end
    end
%{ 
    % 檢查指定代理與其他代理的距離約束(完整軌跡/位置共享)
    for j = 1:num_agents
        if adjacency_matrix(agent_index, j) == 1 && agent_index ~= j
            for n = 1:Np
                % 計算 agent_index 和代理 j 之間的預測距離
                %distance_agents = sqrt((x_pred(n, agent_index) - x_pred(n, j))^2 + (y_pred(n, agent_index) - y_pred(n, j))^2);
                distance_agents = sqrt((x_pred(n, agent_index) - x(j))^2 + (y_pred(n, agent_index) - y(j))^2);
                % 碰撞約束：僅對代理 agent_index 施加約束
                c(constraint_index) = 2 * r_safety - distance_agents; 
                constraint_index = constraint_index + 1;
            end
        end
    end
%}  
    % 檢查指定代理與障礙物的距離
    for obs = 1:length(obstacle_radii)
        for n = 1:Np
            % 計算代理 agent_index 與每個障礙物的距離
            distance_obstacle = sqrt((x_pred(n, agent_index) - obstacle_positions(obs, 1))^2 + (y_pred(n, agent_index) - obstacle_positions(obs, 2))^2);
            
            % 計算代理半徑和障礙物半徑之和作為安全距離
            total_safe_distance = r_safety_obstacle + obstacle_radii(obs);
            
            % 障礙物碰撞約束
            c(constraint_index) = total_safe_distance - distance_obstacle;
            constraint_index = constraint_index + 1;
        end
    end

    % 返回硬約束，去除多餘的零元素
    c = c(1:constraint_index-1);
end