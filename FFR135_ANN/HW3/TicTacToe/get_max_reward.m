function [max_val, index_max_val] = get_max_reward(rewards_matrix)

    max_val = nanmax(rewards_matrix,[],"all");
    
    linear_index_max_val = find(rewards_matrix==max_val);
    [x_index, y_index] = ind2sub(size(rewards_matrix),linear_index_max_val);

    index_max_val = [x_index y_index];
end