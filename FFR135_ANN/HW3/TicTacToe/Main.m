clc, clear
tic
% parameters
n_games = 3e4;
board_size = 3; % square board

empty_board = zeros(board_size);
empty_rewards = empty_board;
Q1 = {empty_board;empty_rewards};
Q2 = {empty_board;empty_rewards};

epsilon = 1; % starting values
winners = [];
game_results = [];
update_games = 101:100:n_games; % make updates after each 100 games
for game = 1:n_games
    t = 0; % time step counter, number of moves made
    if any(ismember(game,update_games)) == true 
        epsilon = epsilon*0.94; % reduce exploration
        
        % Count up wins, draws as share of 100 games
        wins_player1 = sum(winners == 1)/100;
        wins_player2 = sum(winners == 2)/100;
        draws = sum(winners == 0)/100;
        game_results = [game_results; wins_player1 wins_player2 draws];

        % reset storing variables
        winners = [];
        wins_player1 = [];
        wins_player2 = [];
        draws = [];
    end

    % initialize boards, one for each players' perspective
    board1 = empty_board;
    board2 = empty_board;
    
    % player 1 moves
    board1_copy = board1; % save for Q1 update
    [move1,Q1] = get_move(1,Q1,board1,epsilon);
    board2(move1(1),move1(2)) = move1(3); % edit board of next player
    board1 = board2; % update board of first player
    t = t + 1;
    
    % player 2 moves
    board2_copy = board2;
    [move2,Q2] = get_move(2,Q2,board2,epsilon);
    board1(move2(1),move2(2)) = move2(3);
    board2 = board1;
    t = t + 1;

    % update Q1
    Q1 = update_table(Q1,board1_copy,board1,move1,0);

    max_iters = 10;
    for iter = 1:max_iters
        board1_copy = board1;
        [move1,Q1] = get_move(1,Q1,board1,epsilon);
        board2(move1(1),move1(2)) = move1(3);
        board1 = board2;
        t = t + 1;

        % check if game is over
        [game_over, winner] = check_game_over(board2);
        if game_over == true
            winners = [winners, winner];
            % give out rewards
            if winner == 1
                Q1 = update_table(Q1,board1_copy,board1,move1,1);
                Q2 = update_table(Q2,board2_copy,board2,move2,-1);
            elseif winner == 2
                Q1 = update_table(Q1,board1_copy,board1,move1,-1);
                Q2 = update_table(Q2,board2_copy,board2,move2,1);
            else % draw
                Q1 = update_table(Q1,board1_copy,board1,move1,0);
                Q2 = update_table(Q2,board2_copy,board2,move2,0);
            end
            final_board = board2;
            break;
        end
        Q2 = update_table(Q2,board2_copy,board2,move2,0);

        board2_copy = board2;
        [move2,Q2] = get_move(2,Q2,board2,epsilon);
        board1(move2(1),move2(2)) = move2(3);
        board2 = board1;
        t = t + 1;
        
        % check if game is over
        [game_over, winner] = check_game_over(board1);
        if game_over == true
            winners = [winners, winner];
            % give out rewards
            if winner == 1
                Q1 = update_table(Q1,board1_copy,board1,move1,1);
                Q2 = update_table(Q2,board2_copy,board2,move2,-1);
            elseif winner == 2
                Q1 = update_table(Q1,board1_copy,board1,move1,-1);
                Q2 = update_table(Q2,board2_copy,board2,move2,1);
            else % draw
                Q1 = update_table(Q1,board1_copy,board1,move1,0);
                Q2 = update_table(Q2,board2_copy,board2,move2,0);
            end
            final_board = board1;
            break;
        end
        Q1 = update_table(Q1,board1_copy,board1,move1,0); 
    end
    disp("Game " + game + " took " + t + " moves.")
    if winner == 0
        disp("No winner, draw")
    else
        disp("Winner: Player " + winner);
    end
    disp("Final board:")
    disp(final_board)
end
plot(update_games,game_results(:,1),update_games,game_results(:,2),update_games,game_results(:,3));
legend("P_1 wins","P_2 wins","Draw","Location","best")
xlabel("rounds played"), ylabel("frequency")

player1 = cell2mat(Q1);
player2 = cell2mat(Q2);
writematrix(player1,"player1.csv")
writematrix(player2,"player2.csv")

runtime = toc;
disp(n_games + " games took " + runtime/60 + " minutes.")