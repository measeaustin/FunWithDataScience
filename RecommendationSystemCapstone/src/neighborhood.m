function [ id_most_sim_usr, coeff_most_sim_usr ] = neighborhood( train_data, test_data, K, L )
    % neighborhood
    % 
    % Description:
    %   Neighborhood method for memory based collaborative filtering. Finds
    %   the L most similar users from the training data for each user in
    %   the test data. Users from the training data are only considered if
    %   they have K item ratings in common with a target user from the test
    %   data. Uses Pearson Correlation measure to compute similarity.
    %
    % Parameters:
    %   train_data: number of training users x m matrix of ratings
    %   test_data:  number of testing users x m matrix of ratings
    %   K:          minimum common item ratings between a training user 
    %               and target user 
    %   L:          minimum number of users needed for the weighted average
    %
    % Output: 
    %   id_most_sim_usr:    user ids for the L most similar users from the
    %                       training set to the target users from testing.
    %                       number of test users x L matrix 
    %   coeff_most_sim_usr: similarity measures for each of the L most
    %                       similar users. number of test users x L matrix
    
    [n_train, ~] = size(train_data);
    [n_test, ~] = size(test_data);
    
    id_most_sim_usr = zeros(n_test,L); % ids for most L similar users
    coeff_most_sim_usr = zeros(n_test,L);% Pearson correlation for each L similar users
    for i  = 1:n_test  % i is the target user id from testing
        usr_test = test_data(i, :);
        % item ids for the items that the test user has rated
        [~, test_obj_id] = find(usr_test);
        
        tic % time each iteration 
        similarity  = zeros(1,n_train);
        for j = 1:n_train % j is the user id from training        
            usr_train = train_data(j, :);
            % item ids for the items that the training user has rated
            [~, train_obj_id] = find(usr_train);
            
            % -------------------------------------------------------------
            % TODO:
            % 1. Use intersect() function to find the items that both the 
            % test and training user have rated. You can find the item ids 
            % that the test user has rated in the test_obj_id vector. You 
            % can find the item ids that the training user has rated in the 
            % train_obj_id vector.
            %
            % 2. If the two users have less than k ratings in common, 
            % then do not consider the training user in the similarity 
            % calculation.
            %
            % 3. Otherwise, compute the Pearson correlation coefficient of
            % both user's common ratings using the function corr()
            %
            % 4. Place the results into similarity(j)
            % -------------------------------------------------------------
            % PLACE YOUR CODE HERE
        end
        
        similarity(isnan(similarity)) = 0; % corr can return NaN sometimes
        % Find the top L similar users and their similarity coefficients
        [sim_sorted, sim_usr_ids] = sort(similarity,'descend');
        id_most_sim_usr(i,:) = sim_usr_ids(1:L); % user index into training data
        coeff_most_sim_usr(i,:) = sim_sorted(1:L); % similarity values for each user
        toc
    end
end

