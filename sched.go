
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package gomas

import (
    "math/rand"
    "sync/atomic"
    "time"
    //"fmt"
)


// ComputeTask is unit of work that is holding other tasks from being run. Argument
// channel is used to indicate worker routine when task is willing to yield to
// other tasks, normally at the end of the function.
type ComputeTask func(chan int) 

// Simple task; function to be executed and channel where to signal
// when it completes.
type Task struct {
    // task id
    id int64
    // who is working
    workerId int64
    // work to do
    work ComputeTask
    // where to tell when ready
    ready chan int
    // next task in chain
    next *Task
}

// Create new task; 
func NewTask(todo ComputeTask, chn chan int) *Task {
    t := new(Task)
    t.work = todo
    t.ready = chn
    return t
}

// safely send Task 
func sendTask(q chan *Task, t *Task) {
    for cap(q) > 0 && len(q) == cap(q) {
        time.Sleep(time.Millisecond)
    }
    q <- t
}

// safely send ok signal
func sendOK(q chan int) {
    for cap(q) > 0 && len(q) == cap(q) {
        time.Sleep(time.Millisecond)
    }
    q <- 1
}

// Worker to execute task on a CPU. 
type Worker struct {
    id int64
    // CPU working executing tasks
    queue chan *Task
    // my scheduler
    sched *Scheduler
    // how many done
    nexec int64
    // how many totally
    ntotal int64
}

func NewWorker(id int64, ntask int, s *Scheduler) *Worker {
    worker := new(Worker)
    worker.id = id
    worker.queue = make(chan *Task, ntask)
    worker.sched = s
    return worker
}

func (w *Worker) Start() {
    go func() {
        w.Run()
    }()
}

func (w *Worker) Stop() {
    w.queue <- nil
}

// Run worker. Waits for tasks on channel. When receives one tries to reserve it. If succesfull
// executes the work function and signal readiness after function returned. If not able to
// reserve forgets the task completely.
func (w *Worker) Run() {
    var t *Task
    var yield chan int = make(chan int)
    for true {
        t = <- w.queue
        if t == nil {
            // stop on nil task
            break
        }
        w.ntotal += 1
        if atomic.CompareAndSwapInt64(&t.workerId, 0, w.id) {
            go func() {
                t.work(yield)
                sendOK(t.ready)
            }()
            // wait for current task to yield
            <- yield
            w.nexec += 1

            if t.next != nil {
                tn := t.next
                // schedule next task in task chain
                go func() {
                    w.sched.Schedule(tn)
                }()
                t.next = nil
            }
        }

    }
    sendOK(w.sched.done)
}

// Statistics for worker: #scheduled, #executed
func (w *Worker) Stats() (int64, int64) {
    return w.ntotal, w.nexec
}

// Send task `t` to worker.
func (w *Worker) SendTo(t *Task) {
    sendTask(w.queue, t)
}

// 
type Scheduler struct {
    ntask int64            // task counter
    inqueue chan *Task     // incoming tasks
    workers []*Worker      // available workers
    done chan int          // channel for workers to signal finishing
    running bool
    rrsched bool           // round-robin sceduling
}

// Create new scheduler with ncpu workers
func NewScheduler(nworker int) *Scheduler {
    sched := new(Scheduler)
    sched.inqueue = make(chan *Task, 20)
    sched.workers = make([]*Worker, nworker)
    for k, _ := range sched.workers {
        sched.workers[k] = NewWorker(int64(k+1), 20, sched)
    }
    sched.done = make(chan int, 1)
    sched.running = false
    sched.rrsched = false
    return sched
}

func (s *Scheduler) NumTask() int64 {
    return s.ntask
}

func (s *Scheduler) Workers() []*Worker {
    return s.workers
}

// Start workers and scheduler.
func (s *Scheduler) Start() {
    if s.running {
        return
    }
    for _, w := range s.workers {
        w.Start()
    }
    go func() {
        s.Run()
    }()
}

// Indicate stopping and wait for workers to stop.
func (s *Scheduler) Stop() {
    if ! s.running {
        return
    }
    sendTask(s.inqueue, nil)
    for k := len(s.workers); k > 0; k-- {
        <- s.done
    }
}

// Schedule task `t` for execution.
func (s *Scheduler) Schedule(t *Task) {
    if s.running == false {
        s.Start()
    }
    s.ntask += 1
    t.id = s.ntask
    sendTask(s.inqueue, t)
}


// Schedule incoming tasks. Randomly selects 2 workers and inserts new task to both
// work queus. 
func (s *Scheduler) Run() {
    var t *Task
    s.running = true

    irr := 0  // round-robin index
    for true {
        t = <- s.inqueue
        if t == nil {
            break
        }

        if len(s.workers) == 1 {
            sendTask(s.workers[0].queue, t)
        } else if s.rrsched || len(s.workers) == 2 {
            // round-robin scheduling
            sendTask(s.workers[irr].queue, t)
            irr = (irr + 1) % len(s.workers)
            sendTask(s.workers[irr].queue, t)
            irr = (irr + 1) % len(s.workers)
        } else {
            var k, j int32
            // random scheduling
            k = rand.Int31n(int32(len(s.workers)))
            for j = k; j == k; j = rand.Int31n(int32(len(s.workers))) {}
            sendTask(s.workers[k].queue, t)
            sendTask(s.workers[j].queue, t)
        }
    }
    // stopping
    for _, w := range s.workers {
        w.Stop()
    }
    s.running = false
}

var __scheduler *Scheduler = nil

// Return current scheduler; if argument list non-empty then return
// first scheduler in list. Otherwise return default scheduler.
func CurrentScheduler(sched... *Scheduler) *Scheduler {
    if len(sched) > 0 {
        return sched[0]
    }
    return __scheduler
}

func createDefaultScheduler(ncpu int, rrsched bool) {
    __scheduler = NewScheduler(ncpu)
    __scheduler.rrsched = rrsched
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
