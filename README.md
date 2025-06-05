#include <bits/stdc++.h>
using namespace std;

// 1. Naive Pattern Matching
void naiveSearch(string txt, string pat) {
    int n = txt.size(), m = pat.size();
    for (int i = 0; i <= n - m; i++) {
        int j = 0;
        while (j < m && txt[i + j] == pat[j]) j++;
        if (j == m) cout << "Pattern found at index " << i << endl;
    }
}
int main_naive() {
    string txt = "AABAACAADAABAABA", pat = "AABA";
    naiveSearch(txt, pat);
    return 0;
}

// 2. KMP Algorithm
vector<int> computeLPS(string pat) {
    int m = pat.size();
    vector<int> lps(m);
    int len = 0, i = 1;
    lps[0] = 0;
    while (i < m) {
        if (pat[i] == pat[len]) lps[i++] = ++len;
        else if (len != 0) len = lps[len - 1];
        else lps[i++] = 0;
    }
    return lps;
}
void KMPSearch(string txt, string pat) {
    int n = txt.size(), m = pat.size();
    vector<int> lps = computeLPS(pat);
    int i = 0, j = 0;
    while (i < n) {
        if (txt[i] == pat[j]) { i++; j++; }
        if (j == m) {
            cout << "Pattern found at index " << i - j << endl;
            j = lps[j - 1];
        } else if (i < n && txt[i] != pat[j]) {
            if (j != 0) j = lps[j - 1];
            else i++;
        }
    }
}
int main_kmp() {
    string txt = "ABABDABACDABABCABAB", pat = "ABABCABAB";
    KMPSearch(txt, pat);
    return 0;
}

// 3. Rabin-Karp Algorithm
void rabinKarp(string txt, string pat, int q) {
    int d = 256;
    int n = txt.size(), m = pat.size();
    int h = 1, p = 0, t = 0;
    for (int i = 0; i < m - 1; i++) h = (h * d) % q;
    for (int i = 0; i < m; i++) {
        p = (d * p + pat[i]) % q;
        t = (d * t + txt[i]) % q;
    }
    for (int i = 0; i <= n - m; i++) {
        if (p == t) {
            int j = 0;
            while (j < m && txt[i + j] == pat[j]) j++;
            if (j == m) cout << "Pattern found at index " << i << endl;
        }
        if (i < n - m) {
            t = (d * (t - txt[i] * h) + txt[i + m]) % q;
            if (t < 0) t += q;
        }
    }
}
int main_rk() {
    string txt = "GEEKS FOR GEEKS", pat = "GEEK";
    rabinKarp(txt, pat, 101);
    return 0;
}

// 4. Travelling Salesman Problem (Brute Force)
int tspCost(vector<vector<int>> &graph, vector<int> &path) {
    int cost = 0;
    for (int i = 0; i < path.size() - 1; i++) cost += graph[path[i]][path[i+1]];
    cost += graph[path.back()][path[0]];
    return cost;
}
int main_tsp() {
    vector<vector<int>> graph = {
        {0,10,15,20},
        {10,0,35,25},
        {15,35,0,30},
        {20,25,30,0}
    };
    vector<int> path = {0,1,2,3};
    int minCost = INT_MAX;
    do {
        minCost = min(minCost, tspCost(graph, path));
    } while (next_permutation(path.begin()+1, path.end()));
    cout << "Minimum Cost: " << minCost << endl;
    return 0;
}

// 5. Knapsack Problem (Brute Force)
int knapsackBrute(int W, vector<int> &wt, vector<int> &val, int n) {
    if (n == 0 || W == 0) return 0;
    if (wt[n-1] > W) return knapsackBrute(W, wt, val, n-1);
    return max(val[n-1] + knapsackBrute(W - wt[n-1], wt, val, n-1),
               knapsackBrute(W, wt, val, n-1));
}
int main_knapsack_brute() {
    vector<int> val = {60,100,120}, wt = {10,20,30};
    int W = 50;
    cout << "Max Value: " << knapsackBrute(W, wt, val, val.size()) << endl;
    return 0;
}

// 6. Assignment Problem (Brute Force)
int assignmentCost(vector<vector<int>> &cost, vector<int> &assign) {
    int sum = 0;
    for (int i = 0; i < assign.size(); i++) sum += cost[i][assign[i]];
    return sum;
}
int main_assignment_brute() {
    vector<vector<int>> cost = {
        {9,2,7,8},
        {6,4,3,7},
        {5,8,1,8},
        {7,6,9,4}
    };
    vector<int> assign = {0,1,2,3};
    int minCost = INT_MAX;
    do {
        minCost = min(minCost, assignmentCost(cost, assign));
    } while (next_permutation(assign.begin(), assign.end()));
    cout << "Min Assignment Cost: " << minCost << endl;
    return 0;
}

// 7. Knapsack Problem (Dynamic Programming)
int main_knapsack_dp() {
    int W = 50;
    vector<int> val = {60,100,120}, wt = {10,20,30};
    int n = val.size();
    vector<vector<int>> dp(n+1, vector<int>(W+1,0));
    for (int i=1; i<=n; i++) {
        for (int w=0; w<=W; w++) {
            if (wt[i-1] <= w) dp[i][w] = max(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w]);
            else dp[i][w] = dp[i-1][w];
        }
    }
    cout << "Max Value: " << dp[n][W] << endl;
    return 0;
}

// 8. Longest Common Subsequence
int main_lcs() {
    string X = "AGGTAB", Y = "GXTXAYB";
    int m = X.size(), n = Y.size();
    vector<vector<int>> L(m+1, vector<int>(n+1,0));
    for (int i=1; i<=m; i++) {
        for (int j=1; j<=n; j++) {
            if (X[i-1] == Y[j-1]) L[i][j] = L[i-1][j-1] + 1;
            else L[i][j] = max(L[i-1][j], L[i][j-1]);
        }
    }
    cout << "LCS length: " << L[m][n] << endl;
    return 0;
}

// 9. Minimum Coin Change (DP)
int main_coin_change() {
    vector<int> coins = {1,2,5};
    int V = 11;
    vector<int> dp(V+1, INT_MAX);
    dp[0] = 0;
    for (int i=1; i<=V; i++) {
        for (int c : coins) {
            if (i >= c && dp[i-c] != INT_MAX) dp[i] = min(dp[i], dp[i-c] + 1);
        }
    }
    cout << "Minimum coins required: " << dp[V] << endl;
    return 0;
}

// 10. Activity Selection Problem (Greedy)
int main_activity_selection() {
    vector<pair<int,int>> acts = {{1,2},{3,4},{0,6},{5,7},{8,9},{5,9}};
    sort(acts.begin(), acts.end(), [](auto &a, auto &b){ return a.second < b.second; });
    int count = 1, last_end = acts[0].second;
    for (int i=1; i<(int)acts.size(); i++) {
        if (acts[i].first >= last_end) {
            count++;
            last_end = acts[i].second;
        }
    }
    cout << "Maximum activities: " << count << endl;
    return 0;
}

// 11. Huffman Coding
struct Node {
    char c; int freq;
    Node *left, *right;
    Node(char c, int freq) : c(c), freq(freq), left(nullptr), right(nullptr) {}
};
struct cmp {
    bool operator()(Node* a, Node* b) { return a->freq > b->freq; }
};
void printCodes(Node* root, string str) {
    if (!root) return;
    if (!root->left && !root->right) cout << root->c << ": " << str << "\n";
    printCodes(root->left, str + "0");
    printCodes(root->right, str + "1");
}
int main_huffman() {
    vector<char> chars = {'a','b','c','d','e','f'};
    vector<int> freq = {5,9,12,13,16,45};
    priority_queue<Node*, vector<Node*>, cmp> pq;
    for (int i=0; i<(int)chars.size(); i++) pq.push(new Node(chars[i], freq[i]));
    while (pq.size() > 1) {
        Node *left = pq.top(); pq.pop();
        Node *right = pq.top(); pq.pop();
        Node *top = new Node('$', left->freq + right->freq);
        top->left = left; top->right = right;
        pq.push(top);
    }
    printCodes(pq.top(), "");
    return 0;
}

// 12. Sieve of Sundaram
int main_sieve_sundaram() {
    int n = 50;
    int k = (n - 1)/2;
    vector<bool> marked(k+1, false);
    for (int i=1; i<=k; i++) {
        for (int j=i; i+j+2*i*j <= k; j++) {
            marked[i+j+2*i*j] = true;
        }
    }
    cout << "Primes up to " << n << ": 2 ";
    for (int i=1; i<=k; i++) {
        if (!marked[i]) cout << 2*i + 1 << " ";
    }
    cout << "\n";
    return 0;
}

// 13. Hamiltonian Circuit (Backtracking)
bool isSafe(int v, vector<vector<int>> &graph, vector<int> &path, int pos) {
    if (!graph[path[pos-1]][v]) return false;
    for (int i=0; i<pos; i++)
        if (path[i] == v) return false;
    return true;
}
bool hamCycleUtil(vector<vector<int>> &graph, vector<int> &path, int pos) {
    if (pos == graph.size()) return graph[path[pos-1]][path[0]];
    for (int v=1; v<graph.size(); v++) {
        if (isSafe(v, graph, path, pos)) {
            path[pos] = v;
            if (hamCycleUtil(graph, path, pos+1)) return true;
            path[pos] = -1;
        }
    }
    return false;
}
int main_hamiltonian() {
    vector<vector<int>> graph = {
        {0,1,0,1,0},
        {1,0,1,1,1},
        {0,1,0,0,1},
        {1,1,0,0,1},
        {0,1,1,1,0}
    };
    vector<int> path(graph.size(), -1);
    path[0] = 0;
    if (hamCycleUtil(graph, path, 1)) {
        cout << "Hamiltonian Cycle: ";
        for (int v : path) cout << v << " ";
        cout << path[0] << "\n";
    } else cout << "No Hamiltonian cycle found\n";
    return 0;
}

// 14. Subset Sum Problem (DP)
bool subsetSum(vector<int> &set, int sum) {
    int n = set.size();
    vector<vector<bool>> dp(n+1, vector<bool>(sum+1, false));
    for (int i=0; i<=n; i++) dp[i][0] = true;
    for (int i=1; i<=n; i++) {
        for (int j=1; j<=sum; j++) {
            if (set[i-1] > j) dp[i][j] = dp[i-1][j];
            else dp[i][j] = dp[i-1][j] || dp[i-1][j - set[i-1]];
        }
    }
    return dp[n][sum];
}
int main_subset_sum() {
    vector<int> set = {3, 34, 4, 12, 5, 2};
    int sum = 9;
    cout << (subsetSum(set, sum) ? "Subset with given sum found\n" : "No subset found\n");
    return 0;
}

// 15. Knight's Tour (Backtracking)
bool isSafeKT(int x, int y, vector<vector<int>> &sol) {
    return (x >= 0 && y >= 0 && x < 8 && y < 8 && sol[x][y] == -1);
}
bool solveKTUtil(int x, int y, int movei, vector<vector<int>> &sol, int dx[], int dy[]) {
    if (movei == 64) return true;
    for (int k=0; k<8; k++) {
        int nx = x + dx[k], ny = y + dy[k];
        if (isSafeKT(nx, ny, sol)) {
            sol[nx][ny] = movei;
            if (solveKTUtil(nx, ny, movei+1, sol, dx, dy)) return true;
            sol[nx][ny] = -1;
        }
    }
    return false;
}
int main_knight_tour() {
    vector<vector<int>> sol(8, vector<int>(8, -1));
    int dx[8] = {2,1,-1,-2,-2,-1,1,2};
    int dy[8] = {1,2,2,1,-1,-2,-2,-1};
    sol[0][0] = 0;
    if (solveKTUtil(0,0,1,sol,dx,dy)) {
        for (auto &row : sol) {
            for (int val : row) cout << val << " ";
            cout << "\n";
        }
    } else cout << "No solution found\n";
    return 0;
}

// 16. Sudoku Solver (Backtracking)
bool isValidSudoku(vector<vector<int>> &grid, int r, int c, int num) {
    for (int i=0; i<9; i++) {
        if (grid[r][i] == num) return false;
        if (grid[i][c] == num) return false;
        if (grid[3*(r/3) + i/3][3*(c/3) + i%3] == num) return false;
    }
    return true;
}
bool solveSudokuUtil(vector<vector<int>> &grid) {
    for (int r=0; r<9; r++) {
        for (int c=0; c<9; c++) {
            if (grid[r][c] == 0) {
                for (int num=1; num<=9; num++) {
                    if (isValidSudoku(grid, r, c, num)) {
                        grid[r][c] = num;
                        if (solveSudokuUtil(grid)) return true;
                        grid[r][c] = 0;
                    }
                }
                return false;
            }
        }
    }
    return true;
}
int main_sudoku() {
    vector<vector<int>> grid = {
        {5,3,0,0,7,0,0,0,0},
        {6,0,0,1,9,5,0,0,0},
        {0,9,8,0,0,0,0,6,0},
        {8,0,0,0,6,0,0,0,3},
        {4,0,0,8,0,3,0,0,1},
        {7,0,0,0,2,0,0,0,6},
        {0,6,0,0,0,0,2,8,0},
        {0,0,0,4,1,9,0,0,5},
        {0,0,0,0,8,0,0,7,9}
    };
    if (solveSudokuUtil(grid)) {
        for (auto &row : grid) {
            for (int num : row) cout << num << " ";
            cout << "\n";
        }
    } else cout << "No solution exists\n";
    return 0;
}
