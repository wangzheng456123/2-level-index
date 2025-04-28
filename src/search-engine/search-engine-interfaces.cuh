#include <core/interfaces.cuh>
#include <utils/config_utils.h>

class search_engine_base {
public:
    search_engine_base() {}
    virtual ~search_engine_base() = default;
    virtual void run() = 0;
protected:
private:
};