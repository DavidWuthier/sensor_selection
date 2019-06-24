classdef fun
    
    methods (Static)
        
        function v = quadratic(x,par)
           v = (1/2)*x'*par.P*x + par.q'*x + par.r;
        end
        
        function g = quadratic_grad(x,par)
           g = par.P*x + par.q;
        end
        
        function h = quadratic_hess(x,par)
           h = par.P;
        end
        
        function v = LogVolume(z,par)
            cov=0;
            for i=1:par.m
                cov=cov+z(i)*(par.a(:,i)*par.a(:,i)');
            end
            
            v = log(det(cov));
        end
        
        function cov = cov(z,par)
            cov=0;
            for i=1:par.m
                cov=cov+z(i)*(par.a(:,i)*par.a(:,i)');
            end
        end
        
        function v = ApproxLogVolume(z,par)
            cov=fun.cov(z,par);
            
            if cov==0
                v=-10e6; % if cov is singular;
            else
            v = log(det(cov)) ...
              + par.kappa*sum(log(z) + log(1-z),1);
            end
        end            
        
        function g = ApproxLogVolume_grad(z,par)
            cov=fun.cov(z,par);
            L = chol(cov,'upper'); % Cholesky factoriazation;
            
            g=NaN(par.m,1);
            for i=1:par.m
                g(i) = par.a(:,i)'*(L\(L'\par.a(:,i))) ... 
                     + par.kappa/z(i) - par.kappa/(1-z(i));
            end
            
        end
        
        function h = ApproxLogVolume_hess(z,par)
            cov=fun.cov(z,par);
            L = chol(cov,'upper'); % Cholesky factoriazation;
            
            d = 1/(z.^2) + 1/((1-z).^2);
            
            h = par.a'*(L\(L'\par.a)).^2 ...
                - par.kappa*diag(d);
            
        end
        
    end
end

