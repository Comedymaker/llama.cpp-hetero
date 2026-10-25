#include "llama.h"
#include "llama-ext.h"
#include "verified-private-api.h"
#include <algorithm>
#include <map>
#include "slot-pool.h"
static std::map<llama_context*,int> slots;
static unsigned next_slot=0;
static bool prepare_layer(llama_context *ctx){
 int slot=next_slot++%2;slots[ctx]=slot;slot_pool::scope scope(slot);
 return llama_prepare_layer_weights_direct(ctx);
}
static int decode_layer(llama_context *ctx,llama_batch batch){
 auto it=slots.find(ctx);slot_pool::scope scope(it==slots.end()?-1:it->second);
 return llama_decode(ctx,batch);
}
#define llama_prepare_layer_weights_direct prepare_layer
#define llama_decode decode_layer
#define main original_layerskip_main
#include "queue.cpp"
#undef main
int main(int argc, char ** argv) {
 if (argc == 2 && !std::strcmp(argv[1], "--batch-unit")) {
  if(!batch_gain_unit_tests())return 1;
  if(planned_initial().enabled){
   gain_tree_cost c{{},1000.,.1,1.};
   if(c.estimate(12,3)!=planned_initial().verification(12))return 1;
   c.observe(12,3,900.);c.observe(12,3,800.);
   if(std::abs(c.estimate(12,3)-880.)>1e-6)return 1;
   double expected=880.+planned_initial().verification(13)-planned_initial().verification(12);
   if(std::abs(c.estimate(13,3)-expected)>1e-6)return 1;
  }
  return 0;
 }
 return original_layerskip_main(argc, argv);
}
