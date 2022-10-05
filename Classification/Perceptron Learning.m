clear plot;

%2 Input Perceptron Learning Classification

p = [1 2 3 1 2 4; 4 5 3.5 0.5 2 0.5]; 
t = [1 1 1 0 0 0];
W = [0 0];
b = 0;
N = 6; % Number of iterations = Number of inputs

flags = ones(1,N); % Flag vector keeps track of inputs that have been classified correctly. 1 = classified incorrectly, 0 = classified correctly. One vector value for each input.


while flag_count(flags, N) == 1 % Call function flag_count and check for flag_check.
    % While at least one flag = 1, [flag_check=1] (Incomplete classification, need all flags in flags vector = 0). 

    flags = ones(1, N); % Will reset flags to [1 1 1 1 1 1] to test each of them again.

    for k = 1:N % Loop through each input p1:pN.
        p_k = p(:,k); % Select input vector p(k) from p matrix.
        a_k = hardlim(W*p_k + b); % Calculate output.
        e_k = t(k) - a_k; % Calculate error.

        if e_k == 0 % If error(k) = 0.
            flags(k) = 0; % Set input vector flag(k) to 0. Classified correctly. 
        else
            W(1) = W(1) + e_k*p_k(1); % Update weight column 1 in W vector.
            W(2) = W(2) + e_k*p_k(2); % Update weight column 2 in W vector.
            b = b + e_k; % Update scalar bias.
        end
    end
end

W % Print final weight value for inspection.
b % Print final bias value for inspection.

x = -1:0.1:6; % X-axis range.
y = ((-W(1))/(W(2)))*x - b/(W(2)); % Decision boundary equation W(1)*x+W(2)*y+b=0. Calculating y over x range [-1, 6].

plot(x,y, 'k', LineWidth=0.8); % Plot DB LINE
xlabel('p1');
ylabel('p2');
title('Perceptron Learning Classification')
hold on;

% Print DB equation y = c*x + d or p2 = c*p1 + d. For inspection.
c_str = num2str(-W(1)/W(2)); % c
d_str = num2str(-b/W(2)); % d
decision_boundary_equation = append('p2 = ', c_str, '*p1 + ', d_str ) % y = cx + b

% To plot input points.
for k=1:N
    p_k = p(:,k); % Select first input p(k) from p matrix.
    if t(k) == 1; % If class 1 (t=1)
        scatter(p_k(1), p_k(2), 'filled', 'm'); % Plot point in magenta.
    elseif t(k) == 0; % If class 2 (t=0)
        scatter(p_k(1), p_k(2), 'filled', 'b'); % Plot point in blue.
    end
end


function [flag_check] = flag_count(flags, N) % Function goes through every flag and checks that the inputs were succesfully classified. 
    flag_check = 0; % Set as correctly classified 
    for i = 1:N % Loop thorugh every flag in flags vector.
        if flags(i) ~= 0 % If at least one flag does not equal zero:
            flag_check = 1; % Classify flag as incorrectly.
        end
    end
end

