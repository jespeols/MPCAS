function new_table = update_table(player_table, update_board, new_board, move, reward)

    alpha = 0.1;
    gamma = 1;
    R = reward;

    % get indices in player_table
    update_board_index = get_board_index(player_table,update_board);
    new_board_index = get_board_index(player_table,new_board);
    
    if new_board_index == 0 
        player_table{1,end+1} = new_board; % adds the current board to list of states
        player_table{2,end} = initialize_rewards(new_board);
        new_board_index = size(player_table,2);
    end

        % retrieve corresponding rewards matrices
        update_rewards = player_table{2,update_board_index};

        is_new_board_full = ~any(new_board==0);
        if is_new_board_full == true % rewards matrix is all NaN, so set max reward to zero
            max_val = 0;        
        else 
            rewards_new = player_table{2,new_board_index};           
            % find max value for new board
            [max_val, ~] = get_max_reward(rewards_new);
        end

        % update rewards matrix
        update_val = alpha*(R + gamma*max_val - update_rewards(move(1),move(2)));
        update_rewards(move(1),move(2)) = update_rewards(move(1),move(2)) + update_val;
        
        player_table{2,update_board_index} = update_rewards;
        new_table = player_table;
end