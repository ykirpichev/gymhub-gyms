#include <list>
#include <unordered_map>
#include <utility>
using namespace std;

class LRUCache {
  public:
    LRUCache(int capacity) : capacity(capacity) { where.reserve(capacity + 1); }

    int get(int key) {
        auto it = where.find(key);
        if (it == where.end()) return -1;
        order.splice(order.begin(), order, it->second);  // move to front, O(1)
        return it->second->second;
    }

    void put(int key, int value) {
        auto it = where.find(key);
        if (it != where.end()) {
            it->second->second = value;
            order.splice(order.begin(), order, it->second);
            return;
        }
        order.emplace_front(key, value);
        where[key] = order.begin();
        if ((int)where.size() > capacity) {
            where.erase(order.back().first);
            order.pop_back();
        }
    }

  private:
    int capacity;
    list<pair<int, int>> order;  // most recent first
    unordered_map<int, list<pair<int, int>>::iterator> where;
};
