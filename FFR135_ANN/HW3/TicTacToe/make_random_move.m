function rand_move = make_random_move(current_board)

    empty_positions = find(current_board==0);
    rand_position = empty_positions(randi(length(empty_positions)));
    [x_move, y_move] = ind2sub(3,rand_position);
    rand_move = [x_move y_move];
end