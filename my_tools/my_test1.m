function [Z,E,F,S,history] = my_test1( X, nC, M, beta, mi, k_np, lambda2, betaf )
%  % X为输入，nC为类别数，M为锚点数
% beta张量的权重，mi也是张量里的参数
% k_np为计算gama的参数
% lambda1为12范数项的系数
% lambda2为张量项的系数
% betaf为tr（~）项的系数 此处显示详细说明
if nargin < 4
    if M<6
        k_np = nC-1;
    else
        k_np = 5;
    end
end
nV = length(X); N = size(X{1},1);
for v=1:nV
    S{v}=zeros(N,M); %Get temporary matrix
    W{v}=zeros(N,M);
    J{v}=zeros(N,M);
    E{v}=zeros(N,M);
    Y2{v}=zeros(N,M);
    Z{v}=zeros(N,M);
end
F = zeros(N+M,nC); sX = [N, M, nV];
Isconverg = 0; epson = 10e-5; iter=0;
mu = 10e-5; max_mu = 10e10; pho_mu = 1.1;
rho = 0.0001; max_rho = 10e12; pho_rho = 1.1; %Initialize some possible parameters

disp('----------Anchor Selection----------');

tic;
opt1. style = 1;
opt1. IterMax =50;
opt1. toy = 0;

%%第一步骤初始化 Av，S
[centers, B] = FastmultiCLR(X,nC,0.7,opt1,10);
S=B;
for v=1:nV
    A{v} = centers{v}';
end
toc

num = size(X{1,1},1);
gama = zeros(nV);

%%%calculate adaptive parameters game(v)
for v = 1:nV
    distX = L2_distance_1( X{v}',X{v}' );
    [distXs, ~] = sort(distX,2);
    rr = zeros(num,1);
    for i = 1:num
        di = distXs(i,2:k_np+2);
        rr(i) = 0.5*(k_np*di(k_np+1)-sum(di(1:k_np)));
    end
    gama(v) = mean(rr);
end
sq_w = nV *ones(1,nV);                        
    while(Isconverg == 0)
        %% 1 =========update F^v, 由（2）
        SS = S{1};
        for v = 2: nV
            SS = SS + S{v};
        end
        SS = SS./nV;
        Du = diag(1./sqrt(sum(SS, 2)+ eps));
        Dv = diag(1./sqrt(sum(SS, 1)+ eps));
        [uu, ~, vv] = svd(Du*SS*Dv);
        F = 1/sqrt(2) * [uu(:, 1:nC); vv(:, 1:nC)];
        
        %% 2 =========update J^v，由（4）
        S_tensor = cat(3, S{:,:});
        W_tensor = cat(3, W{:,:});
        s = S_tensor(:);
        w = W_tensor(:);
        [myj, ~] = wshrinkObj_weight_lp(s + 1/rho*w,lambda2 * beta./rho,sX, 0,3,mi);
        J_tensor = reshape(myj, sX);
        W_tensor = reshape(w, sX);
        
        %% 4==========update S^v， 由（3）
        for v = 1:nV
            y = zeros(N, M);
            ff1 = zeros(N, nC);
            ff2 = zeros(M, nC);
            for i = 1:N
                ff1(i,:) = F(i, :)*Du(i, i);
            end
            for j = 1:M
                ff2(j,:) = F(N + j, :)*Dv(j, j);
            end
            L = repmat((sum((ff1).^2, 2)),[1, M]) + repmat((sum((ff2).^2, 2))',[N,1]) - 2*ff1*ff2';
            DD{v} = repmat((sum((X{v}).^2, 2)),[1, M]) + repmat((sum((A{v}).^2, 1)),[N,1]) - 2*X{v}*A{v};
            QQ = J{v} - W{v}/rho;
            L = sort(L,2);
            for i = 1:N
                y(i, :) = DD{v}(i, :)/sqrt(sq_w(v)) - rho*QQ(i, :) + betaf/nV *L(i, :);
                MM = -(y(i, :))/ (rho + 2 * gama(v));%Temporary variables
                S{v}(i, :) = EProjSimplex_new(MM, 1);
            end
        end
        
        %% update W
        for k=1:nV
            J{k} = J_tensor(:,:,k);
            W{k} = W_tensor(:,:,k) + rho*(S{k} - J{k});
        end
        % Update parameters
        mu = min(mu*pho_mu, max_mu);
        rho = min(rho*pho_rho, max_rho);
        old_sqw = sq_w;
        %% 由Sv计算sq_w_v
        for v = 1:nV
            sq_w(v) = sum(sum(DD{v}.*S{v}));
        end
        
        %%  ==============迭代收敛======

        Isconverg = 1;
        if (norm(old_sqw - sq_w)>epson)
            history.norm_sqw(iter+1) = norm(old_sqw - sq_w);
            Isconverg = 0;
        end             
        if (iter>100)
            Isconverg  = 1;
        end
        iter = iter + 1;
        
    end   
end



