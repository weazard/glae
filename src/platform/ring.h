#pragma once
#include <cstdint>

#define RING_SIZE 4096

struct Ring {
    uint8_t  buf[RING_SIZE];
    volatile uint32_t head;  // written by producer
    volatile uint32_t tail;  // written by consumer
};

// Producer (single): returns false if full
static inline __host__ __device__ bool ring_push(Ring* r, uint8_t byte) {
    uint32_t next = (r->head + 1) % RING_SIZE;
    if (next == r->tail) return false;
    r->buf[r->head] = byte;
    r->head = next;
    return true;
}

// Consumer (single): returns false if empty
static inline __host__ __device__ bool ring_pop(Ring* r, uint8_t* byte) {
    if (r->head == r->tail) return false;
    *byte = r->buf[r->tail];
    r->tail = (r->tail + 1) % RING_SIZE;
    return true;
}

static inline __host__ __device__ bool ring_empty(const Ring* r) {
    return r->head == r->tail;
}

static inline __host__ __device__ uint32_t ring_count(const Ring* r) {
    return (r->head - r->tail + RING_SIZE) % RING_SIZE;
}
