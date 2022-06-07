function [S, DD] = solve_S(W,J,S, sq_w, X, A, nC, beta, mi, lambda2, betaf, gama)

M = size(A{1},2);nV = length(X); N = size(X{1},1);
F = zeros(N+M,nC); sX = [N, M, nV];
Isconverg = 0; epson = 10e-10; iter = 0;
mu = 10e-5; max_mu = 10e10; pho_mu = 1.1;
rho = 0.0001; max_rho = 10e12; pho_rho = 1.1;

for iterr = 1:50
    
    %% 1 =========update F^v, 由（2）
    SS = S{1};
    for v = 2: nV
        SS = SS + S{v};
    end
    SS = SS./nV;
    
%     pp = zeros(N+M, N+M);
%     pp(1:N,1+N:N+M) = SS;
%     pp(1+N:N+M,1:N) = SS';
%     DDD = diag(1./sqrt(sum(pp, 2)+ eps));
    Du = diag(1./sqrt(sum(SS, 2)+ eps));
    Dv = diag(1./sqrt(sum(SS, 1)+ eps));%%%%%？
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
    
    %% 2.5 =========update w， w为Y1v
    %         w = w + rho*(s - myj);
    
    %         %% 3 =========update Z^v, Y2^v，由（5）
    %         temp_E =[];
    %         for k =1:nV
    %             temp_E=[temp_E;S{k}+Y2{k}/mu];
    %         end
    %         [Econcat] = solve_l1l2(temp_E,lambda1/mu);
    %         ro_b =0;
    %         ro_end = size(X{1},1);
    %         Z{1} =  Econcat(1:size(X{1},1),:);
    %         for i = 2:nV
    %             ro_b = ro_b + size(X{i-1},1);
    %             ro_end = ro_end + size(X{i},1);
    %             Z{i} =  Econcat(ro_b+1:ro_end,:);
    %             Y2{i} = Y2{i} + mu*(S{i}-Z{i});
    %         end
    
    %% 4==========update S^v， 由（3）
    for v = 1:nV
        y = zeros(N, M);
        % 初始化D,Q,B,L
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
        %%第一个式子形成Nx1的矩阵，在复制成NXM.
        %             for i = 1:N
        %                 for j = 1:M
        %                     sq_w = sq_w + DD(i, j) * S{v}(i, j);
        %                 end
        %             end
        QQ = J{v} - W{v}/rho;
        % B
        
        L = sort(L,2);
        for i = 1:N
            y(i, :) = DD{v}(i, :)/sqrt(sq_w(v)) - rho*QQ(i, :) + betaf/nV *L(i, :);
            MM = -(y(i, :))/ (rho + 2 * gama(v));
            S{v}(i, :) = EProjSimplex_new(MM, 1);
        end
    end
    
    %% J,Y1
    for k=1:nV
        J{k} = J_tensor(:,:,k);
        W{k} = W_tensor(:,:,k) + rho*(S{k} - J{k});
    end
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
%     sq_w = sum(sum(DD.*S{v}));
    %% 5==========update Wv^v， 由老师推导
    %         for v = 1:nV
    %             % 迭代求解
    %             for jj = 1:5
    %                 Bv = rand(N, length(sum(diag(S{v}(1,:))*A{v}')));
    %                 % 第一次迭代
    %                 diag_s = diag(sum(S{v}, 2));
    %                 Hv = X{v}'*diag_s*X{v};
    %                 for i = 1:M
    %                     Bv(i, :) = sum(diag(S{v}(i,:))*A{v}');
    %                 end
    %                 Qv = X{v}'*Bv;
    %                 col_x = size(X{v}, 2);
    %                 Pv = (yy * eye(col_x) - Hv) * Wv{v} + Qv;
    %                 [uq,~,vq] = svd(Pv);
    %                 new_W = uq(:,1:dim) * vq;
    %             end
    %             % 迭代求解之后的结果，作为返回
    %             Wv{v} = new_W;
    %         end
    %                 Wv{v} = new_W;
    %                 diag_s = diag(sum(S{v}, 2));
    %                 Hv = X{v}'*diag_s*X{v}*Wv{v};
    %                 for i = 1:M
    %                     Bv(i, :) = sum(diag(S{v}(i,:))*A{v}');
    %                 end
    %                 Qv = X{v}'*Bv;
    %                 col_x = size(X{v}, 2);
    %                 Pv = yy * eye(col_x) * Wv{v} - Hv + Qv;
    %                 [uq,~,vq] = svd(Pv);
    %                 new_W = uq(:,1:dim) * vq';
end
