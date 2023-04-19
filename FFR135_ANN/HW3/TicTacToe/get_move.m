function [move, new_table] = get_move(player,player_table,board,epsilon)

    move = zeros(1,3);
    if player == 1
        sign = 1;
    else
        sign = -1;
    end
    move(3) = sign;

    % check if board is encountered
    board_index = get_board_index(player_table,board);
    if board_index == 0 % board was not found, add & initialize
        player_table{1,end+1} = board; % adds the current board to list of states
        player_table{2,end} = initialize_rewards(board);
    end
  
    r = rand;
    if r < epsilon % random move
        rand_move = make_random_move(board);
        move(1:2) = rand_move;
    else
        if board_index == 0 % board was not found, move will be random
            rand_move = make_random_move(board);
            move(1:2) = rand_move;
        else
            rewards = player_table{2,board_index};
            % make move based on max estimated future reward
            [~, index_max_val] = get_max_reward(rewards);
            if size(index_max_val,1) > 1 % degenerate moves, choose a move randomly
                r = randi(size(index_max_val,1));
                index_max_val = index_max_val(r,:);
            end
            move(1:2) = index_max_val;
        end
    end
    new_table = player_table;
end