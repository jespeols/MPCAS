function [game_over, winner] = check_game_over(board)

    board_width = size(board,2);
    board_height = size(board,1);
    board_diag = diag(board);
    board_diag2 = diag(fliplr(board));
    diag_length = length(board_diag);
    
    game_over = false; % returned unless win condition is found
    winner = nan;   
    % check columns
    for c = 1:board_width
        col_sum = sum(board(:,c));
        if col_sum == board_width % player 1 wins
            winner = 1;
            game_over = true;
        elseif col_sum == -board_width
            winner = 2;
            game_over = true;
        end
    end

    if game_over == false % check rows
        for r = 1:board_height
            row_sum = sum(board(r,:));
            if row_sum == board_height % player 1 wins
                winner = 1;
                game_over = true;
            elseif row_sum == -board_height
                winner = 2;
                game_over = true;
            end
        end
    end
    
    if game_over == false % check diagonals
        diag_sum = sum(board_diag);
        diag2_sum = sum(board_diag2);
        if diag_sum == diag_length || diag2_sum == diag_length % player 1 wins
            winner = 1;
            game_over = true;
        elseif diag_sum == -diag_length || diag2_sum == -diag_length
            winner = 2;
            game_over = true;
        end
    end
    
    if game_over == false % check for draw
        empty_positions = find(board==0);
        if isempty(empty_positions) == true
            game_over = true;
            winner = 0;
        end
    end
end