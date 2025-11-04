use std::ops::{Index, IndexMut};

struct CircularBuffer<T> {
    buffer: Vec<Option<T>>,
    head: usize,
    tail: usize,
    capacity: usize,
    size: usize,
}

impl<T> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than zero");
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);
        CircularBuffer {
            buffer,
            head: 0,
            tail: 0,
            capacity,
            size: 0,
        }
    }

    /// Number of elements currently stored.
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }


    pub fn push(&mut self, item: T) {
        self.buffer[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;

        if self.size == self.capacity {
            // overwritten the oldest; advance tail to the new oldest
            self.tail = self.head;
        } else {
            self.size += 1;
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {

            return None;
        }
        let item = self.buffer[self.tail].take();
        self.tail = (self.tail + 1) % self.capacity;
        self.size -= 1;
        item
    }

    /// Get a reference to an element by relative index:
    /// - 0 => most recent
    /// - -1 => previous
    /// - -k => k steps back (oldest is at index `-(size-1)`)
    pub fn get_rel(&self, idx: isize) -> Option<&T> {
        if idx > 0 {
            return None;
        }
        let k = (-idx) as usize; // 0 => most recent, 1 => previous, ...
        if k >= self.size {
            return None;
        }
        let pos = (self.head + self.capacity - 1 - k) % self.capacity;
        self.buffer[pos].as_ref()
    }

    /// Mutable variant of get_rel
    pub fn get_rel_mut(&mut self, idx: isize) -> Option<&mut T> {
        if idx > 0 {
            return None;
        }
        let k = (-idx) as usize;
        if k >= self.size {
            return None;
        }
        let pos = (self.head + self.capacity - 1 - k) % self.capacity;
        self.buffer[pos].as_mut()
    }

    /// Return a slice of occupied slots as references in most-recent-first order.
    /// This constructs a Vec<&T> for convenience.
    pub fn recent_slice(&self) -> Vec<&T> {
        let mut out = Vec::with_capacity(self.size);
        for k in 0..self.size {
            let pos = (self.head + self.capacity - 1 - k) % self.capacity;
            if let Some(ref v) = self.buffer[pos] {
                out.push(v);
            }
        }
        out
    }

    pub fn recent_iter(&self) -> impl Iterator<Item = &T> {
        let head = self.head;
        let cap = self.capacity;
        let buf = &self.buffer;
        (0..self.size).filter_map(move |k| {
            let pos = (head + cap - 1 - k) % cap;
            buf[pos].as_ref()
        })
    }
}

// Provide panicking indexing like `buf[0]` => most recent, `buf[-1]` => previous.
// Indexing out of range will panic (similar to Vec).
impl<T> Index<isize> for CircularBuffer<T> {
    type Output = T;
    fn index(&self, index: isize) -> &Self::Output {
        self.get_rel(index).expect("circular buffer index out of bounds")
    }
}

impl<T> IndexMut<isize> for CircularBuffer<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        self.get_rel_mut(index).expect("circular buffer index out of bounds")
    }
}

#[cfg(test)]
mod tests {
    use super::CircularBuffer;

#[test]
    fn push_wraps_and_overwrites_oldest() {
        let mut cb = CircularBuffer::new(3);
        assert!(cb.is_empty());
        cb.push(1);
        cb.push(2);
        cb.push(3);
        assert!(cb.is_full());
        // next pushes overwrite oldest entries
        cb.push(4);
        cb.push(5);
        assert_eq!(cb.len(), 3);
        assert_eq!(cb.get_rel(0), Some(&5));
        assert_eq!(cb.get_rel(-1), Some(&4));
        assert_eq!(cb.get_rel(-2), Some(&3));
        assert!(cb.get_rel(-3).is_none());
    }

    #[test]
    fn pop_removes_oldest_in_fifo_order() {
        let mut cb = CircularBuffer::new(3);
        cb.push(10);
        cb.push(20);
        cb.push(30);
        assert_eq!(cb.pop(), Some(10));
        assert_eq!(cb.pop(), Some(20));
        assert_eq!(cb.len(), 1);
        cb.push(40);
        cb.push(50);
        assert_eq!(cb.pop(), Some(30));
        assert_eq!(cb.pop(), Some(40));
        assert_eq!(cb.pop(), Some(50));
        assert!(cb.pop().is_none());
        assert!(cb.is_empty());
    }

    #[test]
    fn get_rel_and_index_access() {
        let mut cb = CircularBuffer::new(4);
        cb.push(7);
        cb.push(8);
        cb.push(9);
        assert_eq!(cb.get_rel(0), Some(&9));
        assert_eq!(cb.get_rel(-1), Some(&8));
        assert_eq!(cb.get_rel(-2), Some(&7));
        assert_eq!(cb[0], 9);
        assert_eq!(cb[-1], 8);
        assert_eq!(cb[-2], 7);
    }

    #[test]
    fn get_rel_mut_allows_mutation() {
        let mut cb = CircularBuffer::new(2);
        cb.push(String::from("a"));
        cb.push(String::from("b"));
        if let Some(slot) = cb.get_rel_mut(-1) {
            slot.push_str("!");
        }
        assert_eq!(cb.get_rel(-1).map(String::as_str), Some("a!"));
        // mutate most recent through IndexMut
        cb[-0].push('~');
        assert_eq!(cb.get_rel(0).map(String::as_str), Some("b~"));
    }

    #[test]
    #[should_panic(expected = "circular buffer index out of bounds")]
    fn index_panics_if_out_of_range() {
        let mut cb = CircularBuffer::new(2);
        cb.push(1);
        let _ = cb[-2];
    }

    #[test]
    #[should_panic(expected = "circular buffer index out of bounds")]
    fn index_mut_panics_if_out_of_range() {
        let mut cb = CircularBuffer::new(2);
        cb.push(1);
        cb.push(2);
        cb[-3] = 99;
    }

    #[test]
    fn recent_slice_returns_most_recent_first() {
        let mut cb = CircularBuffer::new(3);
        assert!(cb.recent_slice().is_empty());
        cb.push(100);
        cb.push(200);
        cb.push(300);
        let snapshot: Vec<i32> = cb.recent_slice().into_iter().copied().collect();
        assert_eq!(snapshot, vec![300, 200, 100]);

        cb.push(400); // overwrite 100
        let snapshot: Vec<i32> = cb.recent_slice().into_iter().copied().collect();
        assert_eq!(snapshot, vec![400, 300, 200]);
    }

        #[test]
    fn recent_iter_yields_most_recent_first() {
        let mut cb = CircularBuffer::new(3);
        cb.push(10);
        cb.push(20);
        cb.push(30);
        let collected: Vec<i32> = cb.recent_iter().copied().collect();
        assert_eq!(collected, vec![30, 20, 10]);

        cb.push(40); // overwrite oldest (10)
        let collected: Vec<i32> = cb.recent_iter().copied().collect();
        assert_eq!(collected, vec![40, 30, 20]);

        // iterator should reflect current len
        assert_eq!(cb.recent_iter().count(), cb.len());
    }
}