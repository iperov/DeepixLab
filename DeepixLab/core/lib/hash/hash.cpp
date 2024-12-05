#include "glsl/glsl.h"
#include "glsl/base_math.h"

extern "C" __declspec(dllexport) void c_similarity_add( uint32_t* similarities, 
                                                        uint8_t* hashed_map,
                                                        uint32_t* hashed_idxs,
                                                        uint64_t* hashes,
                                                        uint32_t& hashed_count,
                                                        uint32_t hash_idx,
                                                        uint64_t hash,
                                                        uint32_t similarity_factor)
{    
    if (hashed_map[hash_idx] == 0)
    {
        hashed_map[hash_idx] = 1;

        uint32_t sim_count = 0;
        for (uint32_t i=0; i<hashed_count; ++i)
        {
            uint32_t sim = bit_count( hashes[i] ^ hash );
            
            if (sim <= similarity_factor)
            {
                similarities[i] += 1;
                ++sim_count;
            }
                
        }
        
        hashed_idxs[hashed_count] = hash_idx;
        hashes[hashed_count] = hash;
        similarities[hash_idx] += sim_count;    
        
        hashed_count += 1;
    }
}


