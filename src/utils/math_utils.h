#pragma once
#include <vector>
#include <functional>

template<typename ElementType, typename IndexType>
int gauss(ElementType a[][5], IndexType n = 4)
{
    int r,c;  
    double eps = 1e-30;
    for(r=0,c=0;c<n;c++)   
    {
        int t=r;
        for(int i=r;i<n;i++)                       
            if(fabs(a[i][c])>fabs(a[t][c]))
                t=i;
        if(fabs(a[t][c])<eps) continue;    
        for(int i=c;i<=n;i++) std::swap(a[t][i],a[r][i]);   
        for(int i=n;i>=c;i--) a[r][i] /=a[r][c];   
        for(int i=r+1;i<n;i++)             
            if(fabs(a[i][c])>eps)
            {
                for(int j=n;j>=c;j--)
                    a[i][j]-=a[i][c]*a[r][j];
            }
        r++;
    }
    if(r<n)     
    {
        for(int i=r;i<n;i++)
            if(fabs(a[i][n])>eps)     
                return 2;
        return 1;        
    }
    for(int i=n-1;i>=0;i--)               
        for(int j=i+1;j<n;j++)
            a[i][n]-=a[j][n]*a[i][j];
    return 0;
}

inline int findMaxFactor(uint64_t a, uint64_t b) {
    if (b <= a) {
        return b; 
    }

    int maxFactor = 1;

    for (int i = 1; i <= sqrt(b); ++i) {
        if (b % i == 0) {
            if (i < a) {
                maxFactor = max(maxFactor, i);
            }
            int pairFactor = b / i;
            if (pairFactor < a) {
                maxFactor = max(maxFactor, pairFactor);
            }
        }
    }

    return maxFactor;
}

inline void uniformly_divide(uint64_t n, uint64_t m, uint64_t i, 
                    uint64_t &size, uint64_t &offset) 
{
    uint64_t r = n % m;
    if (i < r) {
        size = (n + m - 1) / m;
        offset = i * size;
    }
    else {
        size = n / m;
        offset = ((n + m - 1) / m) * r + (i - r) * size; 
    }
}

template<typename T, typename F>
void cartesian_product(const std::vector<std::vector<T>>& S, F&& callback) {
    int k = S.size();
    std::vector<T> current(k);
    int counter = 0;

    std::function<void(int)> dfs = [&](int depth) {
        if (depth == k) {
            callback(current, counter++);
            return;
        }
        for (const auto& val : S[depth]) {
            current[depth] = val;
            dfs(depth + 1);
        }
    };
    dfs(0);
}