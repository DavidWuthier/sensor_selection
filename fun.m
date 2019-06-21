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