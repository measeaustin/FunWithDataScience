function [ q, p, bu, bi, mu ] = sgd_bias( ratings, f, lambda, step_size )
    % sgd_bias
    %
    % Description:
    %   Matrix Factorization for model based collaborative filtering. 
    %   Uses stochastic gradient descent to create a mapping between 
    %   users/items and a latent feature space of f features. Also
    %   computes user and item bias values.
    %
    % Parameters:
    % 	ratings:    a user (n) x item (m) matrix of ratings
    %   f:          number of latent factors
    %   lambda:     regularization weight
    %   step_size:  step size for gradient descent
    %
    % Output:
    %   q:          a f x m item matrix mapped to the latent feature space
    %   p:          a f x n user matrix mapped to the latent feature space
    %   bu:         a 1 x n user bias matrix
    %   bi:         a 1 x m item bias matrix
    %   mu:         mean rating in the ratings matrix
    
    [num_usr, num_obj] = size(ratings);
    num_ratings = nnz(ratings);

    [usr_idx, obj_idx, vals] = find(ratings);

    threshold = 1.0e-3;
    % 30 SGD passes over the data set
    maxiter = 30*num_ratings;

    % Initialize with random values from Gaussian distribution
    q = randn(f,num_obj);
    p = randn(f,num_usr);
    bu = randn(num_usr, 1);
    bi = randn(num_obj, 1);
    mu=mean(vals);
    
    total_err = inf;
    for iter = 0:maxiter
        % t is the index into pick_idx
        t = mod(iter,num_ratings)+1;
        if t == 1
            % Random permutation of ratings
            pick_idx = randperm(num_ratings);

            avg_err = total_err/num_ratings;
            if avg_err < threshold
                break;
            end
            
            disp(['pass is ' num2str(floor(iter/num_ratings))]);
            disp(['average err is ' num2str(avg_err)]);
            total_err = 0;
        end
        % pick a random rating for descent
        idx = pick_idx(t);
        
        u = usr_idx(idx); % u is for user
        i = obj_idx(idx); % i is for item

        r_true = vals(idx); % true rating for user u and item i

        % -----------------------------------------------------------------
        % TODO:
        % Compute the prediction rhat for a particular user u and item i.
        % Refer to the lab overview for the exact formula.
        % -----------------------------------------------------------------
        rhat = ;
        
        err = r_true - rhat;
        total_err = total_err+(err^2);

        % -----------------------------------------------------------------
        % TODO: 
        % Update q_i, p_u, b_u, and b_i using the rules you derived in the
        % lab warm-up
        % -----------------------------------------------------------------
        q(:,i) = ;
        p(:,u) = ;
        bu(u) = ;
        bi(i) = ;
    end
end

