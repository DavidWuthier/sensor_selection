classdef fun
    
    methods (Static)
        
        function v = quadratic(x,par)
           v = (1/2)*x'*par.P*x + par.q'*x + par.r;
        end
        
        function v = quadratic_grad(x,par)
           v = par.P*x + par.q;
        end
        
        function v = quadratic_hess(x,par)
           v = par.P;
        end
        
    end
end
%         function w = compute_w(A,par)
%             W = zeros(par.m,1);
%             for(i=1:par.m)
%                 par.W = par.W + par.z(i) * transpose(A(i,:))*A(i,:);
%             end
%             par.W = par.W^(-1);S
%             
%         end
%         
%         function v = grad_1(A,par)
%             v = zeros(par.m,1);
%             for(i=1:par.m)
%                 v(i) = A(i,:)*par.W*transpose(A(i,:);
%             end
%         end
%         
%         function v = hess_1(A,par)
%             z = zeros(par.m,1);
%             for(i=1:par.m)
%                 z(i) = 1/x(i)^2+1/(1-x(i)^2)
%             end
%             v = -(par.A*par.W*transpose(par.A)).*(par.A*par.W*transpose(par.A))-par.kappa...
%                 .* diag(z)
%     end
%     
% end