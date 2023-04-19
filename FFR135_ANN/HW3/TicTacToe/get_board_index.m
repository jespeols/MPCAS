function [board_index] = get_board_index(player_table, current_board)

    for k = 1:size(player_table,2)
            board = player_table{1,k};
            compare_boards = isequal(board,current_board);
            if compare_boards == true
                board_index = k;
                break;
            end
            if k == size(player_table,2) % board not found
                board_index = 0;
            end            
    end
    
end