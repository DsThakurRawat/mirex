#include <vector>
#include <algorithm>
#include <functional>

using namespace std;

int secureMaximumDeliveries(int deliveryLogs_count, int* deliveryLogs, int k) {
    int max_val = 0;
    for (int i = 0; i < deliveryLogs_count; i++) {
        if (deliveryLogs[i] > max_val) {
            max_val = deliveryLogs[i];
        }
    }
    
    if (max_val == 0) return 0;
    
    long long max_safe_sum = 0;
    int half_k = k / 2;
    
    vector<int> rems(deliveryLogs_count);
    
    for (int M = 1; M <= max_val; M++) {
        long long P = 0; 
        
        for (int i = 0; i < deliveryLogs_count; i++) {
            P += deliveryLogs[i] / M;
            rems[i] = deliveryLogs[i] % M;
        }
        
        if (P < half_k) {
            continue;
        }
        
        long long safe_M_pieces = P - half_k;
        if (safe_M_pieces > half_k) {
            safe_M_pieces = half_k; 
        }
        
        long long current_sum = safe_M_pieces * M;
        int needed_rem = half_k - safe_M_pieces;
        
        if (needed_rem > 0) {
            sort(rems.begin(), rems.end(), greater<int>());
            
            int limit = min(needed_rem, deliveryLogs_count);
            for (int i = 0; i < limit; i++) {
                current_sum += rems[i];
            }
        }
        
        if (current_sum > max_safe_sum) {
            max_safe_sum = current_sum;
        }
    }
    
    return (int)max_safe_sum;
}
