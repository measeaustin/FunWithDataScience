function [ q, p, bu, bi, mu, y ] = svdpp( ratings, itm_rated_4_user, f, lambda, step_size )
    % svdpp
    %
    % Description:
    %   SVD++ algorithm for model based collaborative filtering. 
    %   Uses stochastic gradient descent to create a mapping between 
    %   users/items and a latent feature space of f features. Also
    %   computes user and item bias values.
    %
    % Parameters:
    % 	ratings:            a user (n) x item (m) matrix of ratings
    %   itm_rated_4_user:   a n x m lookup table to get the ids of items that
    %                       a user has provided ratings for. The first cell
    %                       in each row is the number of non-zero entries
    %                       in the row. To get the list of item ids, use
    %                       the code snippet below where u is the user
    %                       id.
    %
    %                       num_itm_rated = itm_rated_4_user(u, 1);
    %                       itm_rated = itm_rated_4_user(u, 2:num_itm_rated+1);
    %
    %   f:                  number of latent factors
    %   lambda:             regularization weight
    %   step_size:          step size for gradient descent
    %
    % Output:
    %   q:          a f x m item matrix mapped to the latent feature space
    %   p:          a f x n user matrix mapped to the latent feature space
    %   bu:         a 1 x n user bias matrix
    %   bi:         a 1 x m item bias matrix
    %   mu:         mean rating in the ratings matrix
    %   y:          a f x m matrix of implicit data values for each latent
    %               feature f of an item
    [num_usr, num_itm] = size(ratings);
    num_ratings = nnz(ratings);

    [usr_idx, itm_idx, vals] = find(ratings);

    threshold = 1.0e-3;
    maxiter = 30*num_ratings;
    
    % Initialize with random values from Gaussian distribution
    q = randn(f,num_itm);
    p = randn(f,num_usr);
    bu = randn(num_usr, 1);
    bi = randn(num_itm, 1);
    y = randn(f,num_itm);
    mu=mean(vals);

    total_err = inf;

    tic
    for iter = 0:maxiter
        % t is the index into pick_idx
        t = mod(iter,num_ratings)+1;
        if t == 1
            toc
            % Random permutation of ratings
            pick_idx = randperm(num_ratings);

            avg_err = total_err/num_ratings;
            if avg_err < threshold
                break;
            end
            
            disp(['pass is ' num2str(floor(iter/num_ratings))]);
            disp(['average err is ' num2str(avg_err)]);
            total_err = 0;
            tic
        end
        % pick a random rating for descent
        idx = pick_idx(t);
        u = usr_idx(idx); % u is for usr
        i = itm_idx(idx); % i is for item

        r_true = vals(idx); % true rating for usr u and item i

        % For speed, get items that a user has rated from lookup table
        num_itm_rated = itm_rated_4_user(u, 1);
        itm_rated = itm_rated_4_user(u, 2:num_itm_rated+1);

        % -----------------------------------------------------------------
        % TODO:
        % Compute the prediction rhat for a particular user u and item i.
        % Refer to the lab overview for the exact formula.
        % For this you will need: 
        % 1. |N(u)| where N(u) is the set of items that user u has rated 
        % 2. The sum of the y_j's where j is an id of an item in N(u). 
        %
        % The vector itm_rated contains the ids of items that a user has 
        % rated (ie. the j's in N(u)) and the variable num_itm_rated
        % contains |N(u)|.
        % -----------------------------------------------------------------
        rhat = ;

        err = r_true - rhat;
        total_err = total_err+(err^2);

        % -----------------------------------------------------------------
        % TODO: 
        % Update q_i, p_u, b_u, and b_i using the rules you derived in the
        % lab warm-up.
        % -----------------------------------------------------------------
        q(:,i) = ;
        p(:,u) = ;
        bu(u) = ;
        bi(i) = ;
        
        % -----------------------------------------------------------------
        % TODO:
        % For all items that a user has rated, update the corresponding 
        % y_j vectors.
        % -----------------------------------------------------------------
        y_itm_rated = y(:, itm_rated); %cache for speed
        y(:,itm_rated) = ;
    end

end

