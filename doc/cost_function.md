
Tracking Cost
$$
J_{tracking}(k)=\sum_{s=1}^{N_{p}} (\sqrt{(p_{l}^x(k+s|k)-p_{t}^x(k+s|k))^2+(p_{l}^y(k+s|k)-p_{t}^y(k+s|k))^2+(p_{l}^z(k+s|k)-p_{t}^z(k+s|k))^2})
$$

Static Object Collision Cost
$$
d(k+s|k,i)=\sqrt{(p_{l}^x(k+s|k)-x_{i})^2+(p_{l}^y(k+s|k)-y_{i})^2+(p_{l}^z(k+s|k)-z_{i})^2}
$$
$$
J_{collision}(k)=\sum_{s=1}^{N_{p}}\sum_{i=1}^{N_{obs}}\frac{Qc}{1+e^{kappa*(d(k+s|k,i)-2*r_{i})}}
$$

Dynamic Object Collision Cost
$$
d(k+s|k,j)=\sqrt{(p_{l}^x(k+s|k)-p_{j}^x(k+s|k))^2+(p_{l}^y(k+s|k)-p_{j}^x(k+s|k))^2+(p_{l}^z(k+s|k)-p_{j}^x(k+s|k))^2}
$$
$$
J_{avoid}(k)=\sum_{s=1}^{N_{p}}\sum_{j=1}^{N_{neighbour}}\frac{Qc}{1+e^{kappa*(d(k+s|k,j)-2*r_{j})}}
$$
