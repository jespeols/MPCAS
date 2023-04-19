function rewards = initialize_rewards(board)
    
    board_size = size(board,1); % assumes square board
    rewards = zeros(board_size);
    for i = 1:board_size
        for j = 1:board_size
            if board(i,j) == -1 || board(i,j) == 1
                rewards(i,j) = nan;
            end
        end
    end

end