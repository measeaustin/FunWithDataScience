load ratings.mat

[num_usr, num_itm] = size(ratings);

% Number of users for testing
n_test = 200;

% Number of users for training
% The line below will use the remaining full dataset for training
% n_train = num_usr - n_test;
n_train = 1000;

% Matrix factorization and SVD++ parameters
f = 20;
lambda = 0.01;
step_size = 0.005;

% Neighborhood method parameters
K = 5;
L = 20;

% split ratings matrix into testing and training datasets
train_idx = randperm(num_usr, n_train);
test_idxidx = randperm(n_test);
test_idxset = setdiff([1:num_usr],train_idx);
test_idx = test_idxset(test_idxidx(1:n_test));

test_data = ratings(test_idx, :);
train_data = ratings(train_idx, :);

% To test predictions, blank out 'perc_blank' percent of the ratings in 
% the testing data set. The remaining '1 - perc_blank' percent of
% the ratings will be included in the full training dataset to establish 
% a baseline for the testing users.
perc_blank = 0.3;
test_data_blanked = zeros(n_test, num_itm);
% the result indicies of the randperm for each user
user2itmratedidx = zeros(n_test, num_itm);
for i = 1:n_test
    u = test_idx(i);
    num_itm_rated = itm_rated_4_user(u, 1);
    itm_rated = itm_rated_4_user(u, 2:num_itm_rated+1);
    
    itm_rated_idx = randperm(num_itm_rated);
    new_len = floor((1.0-perc_blank) * num_itm_rated);
    user2itmratedidx(i, 1:num_itm_rated) = itm_rated_idx;
    itm_rated_idx = itm_rated_idx(1:new_len);
 
    test_data_blanked(i, itm_rated(itm_rated_idx)) = ratings(u, itm_rated(itm_rated_idx));
end

% neighborhood method
[id_sim_usr, coeff_sim_usr] = neighborhood(train_data, test_data_blanked, K, L);

% -------------------------------------------------------------------------
% TODO:
% 1. Using the lookup tables returned from the neighborhood function, find 
% the item rating predictions for each user in the testing set.
%
% 2. Place your results in the neighbor_predictions matrix.
%
% Predictions are the normalized weighted sum of the ratings of the L most
% similar users to a target user in the testing data. See the overview of
% the lab for the formula.
% -------------------------------------------------------------------------
neighbor_predictions = zeros(n_test,num_itm);
% PLACE YOUR CODE HERE


% new training set with the blanked out users
augment_train_data = [test_data_blanked; train_data];

% find matrix factorization and biases using sgd
[q, p, bu, bi, mu] = sgd_bias(augment_train_data, f, lambda, step_size);

% create lookup table of item ids that a user has rated for augmented 
% training set for SVD++ speedup
[~,max_itms_rated] = size(itm_rated_4_user);
lookup_tbl = [zeros(n_test, max_itms_rated); itm_rated_4_user(train_idx, :)];
for i=1:n_test
    [row, col] = find(test_data_blanked(i, :));
     col_size = nnz(test_data_blanked(i, :));
     lookup_tbl(i, 1:(col_size+1)) = [col_size, col];
end
% SVD++
[q_pp, p_pp, bu_pp, bi_pp, mu_pp, y] = svdpp(augment_train_data, lookup_tbl, f, lambda, step_size);

% Compute predictions and errors for each method
total_err_bias = 0;
total_err_neighbor = 0;
total_err_svdpp = 0;
sum_blanked = 0;
sum_neighbor = 0;
for i = 1:n_test % i is the user index in the augmented training matrix
    
    u = test_idx(i); % u is the user index in the original ratings matrix
    num_itm_rated = itm_rated_4_user(u, 1);
    itm_rated = itm_rated_4_user(u, 2:num_itm_rated+1);
    itm_rated_idx = user2itmratedidx(i, 1:num_itm_rated);
    
    % find indicies of blanked out ratings
    new_len = floor((1.0-perc_blank) * num_itm_rated);
    
    % the last 'perc_blank' percent of indexes in itm_rated_idx were
    % blanked out
    blanked_idx = itm_rated(itm_rated_idx(new_len+1:end));
    blanked_len = length(blanked_idx);
    
    % track number of predictions for final error calculation
    sum_blanked = sum_blanked + blanked_len;
    
    actual = ratings(u, blanked_idx);
    
    % evaluate neighborhood method 
    predict_neighbor = neighbor_predictions(i, blanked_idx);
    nonzero_idx = predict_neighbor > 0;
    % only count the ratings that NH method could predict
    % ie. don't count ratings that are zero
    err_neighbor = nansum((predict_neighbor(nonzero_idx) - actual(nonzero_idx)).^2);
    total_err_neighbor = total_err_neighbor + err_neighbor;
    % number of predictions for neighborhood must be kept separately
    sum_neighbor = sum_neighbor + length(nonzero_idx);
    
    % evaluate sgd bias
    bias_sgd = repmat(mu, 1, blanked_len) + repmat(bu(i), 1, blanked_len) + bi(blanked_idx)';
    predict_bias = bias_sgd + (q(:, blanked_idx)'*p(:, i))';
    err_bias = sum((predict_bias - actual).^2);
    total_err_bias = total_err_bias + err_bias;
    
    % evaluate SVD++
    num_filled_itm = lookup_tbl(i,1);
    filled_itm_idx = lookup_tbl(i,2:num_filled_itm+1);
    
    Nu = num_filled_itm^-0.5;
    normalized_yjsum = Nu * sum(y(:, filled_itm_idx), 2);
    bias_pp = repmat(mu_pp, 1, blanked_len) + repmat(bu_pp(i), 1, blanked_len) + bi_pp(blanked_idx)';
    predict_svdpp = bias_pp + (q_pp(:, blanked_idx)'*(p_pp(:, i) + normalized_yjsum))';
    err_svdpp = sum((predict_svdpp - actual).^2);
    total_err_svdpp = total_err_svdpp + err_svdpp;
end

sqrt(total_err_neighbor/sum_neighbor)
sqrt(total_err_bias/sum_blanked)
sqrt(total_err_svdpp/sum_blanked)

