# Parallel Programming Labs

University coursework exploring concurrent and parallel programming concepts at Kaunas University of Technology (KTU).

## Lab 1 - Shared Memory (Bendra atmintis)

Producer-consumer pattern implementation with thread synchronization.

### Lab1a - Monitor-based synchronization (C++)
- Data monitor and result monitor with mutex protection
- Conditional synchronization for blocking when buffer full/empty
- Worker threads process data and filter results
- Results stored in sorted order

### Lab1b - OpenMP (C++)
- Same problem using OpenMP critical sections
- Manual data distribution across threads
- Parallel reduction for computing sums

### Lab1g
- Defense practice exercises

**Concepts:** Mutexes, condition variables, critical sections, producer-consumer pattern

## Lab 2 - Distributed Memory (Paskirstyta atmintis)

Message-passing implementation using Go channels.

- No shared memory — all communication via unbuffered channels
- Four process roles: Main, Data, Result, Workers
- Workers request data, compute results, send to result process
- Implements same filtering and sorting logic as Lab 1

### Lab 2g — Lab 2g-6
- Defense practice exercises with variations

**Concepts:** Channels, goroutines, message passing, distributed architecture

## Tech

- C++ with OpenMP
- Go (goroutines, channels)