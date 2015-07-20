extern crate silinapse;

use std::io::Write;

use silinapse::SymmetricMatrix;
use silinapse::boltzmann::BoltzmannMachine;

// +-------+-------+-------+
// | 5 _ _ | 8 _ 6 | _ _ 4 |
// | _ _ _ | _ _ _ | 8 _ _ |
// | 8 _ 7 | _ 4 _ | _ 5 _ |
// +-------+-------+-------+
// | _ _ 3 | _ 8 _ | 1 9 _ |
// | _ _ _ | 2 _ 4 | _ _ _ |
// | _ 8 6 | _ 5 _ | 4 _ _ |
// +-------+-------+-------+
// | _ 9 _ | _ 7 _ | 2 _ 8 |
// | _ _ 4 | _ _ _ | _ _ _ |
// | 2 _ _ | 9 _ 1 | _ _ 7 |
// +-------+-------+-------+

static INPUT_SUDOKU: [u8; 81] = [
    5, 0, 0, 8, 0, 6, 0, 0, 4,
    0, 0, 0, 0, 0, 0, 8, 0, 0,
    8, 0, 7, 0, 4, 0, 0, 5, 0,
    0, 0, 3, 0, 8, 0, 1, 9, 0,
    0, 0, 0, 2, 0, 4, 0, 0, 0,
    0, 8, 6, 0, 5, 0, 4, 0, 0,
    0, 9, 0, 0, 7, 0, 2, 0, 8,
    0, 0, 4, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 9, 0, 1, 0, 0, 7,
];

static NUM_CHARS: [&'static str; 10] = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",
];

fn sudoku_to_index(c: (u8, u8, u8, u8)) -> usize {
    ((c.0 * 3 + c.2) + 9 * (c.1 * 3 + c.3)) as usize
}

fn display_sudoku(s: &[u8]) {
    assert!(s.len() == 81);
    println!("+-------+-------+-------+");
    for i in 0..81 {
        if i % 9 == 0 { print!("| "); }
        print!("{} ",
            if s[i] == 0 { "_" }
            else if s[i] < 10 { NUM_CHARS[s[i] as usize]}
            else { "X" }
        );
        if i % 3 == 2 { print!("| "); }
        if i % 9 == 8 { println!(""); }
        if i % 27 == 26 {
            println!("+-------+-------+-------+");
        }
    }
}

fn set_values(machine: &mut BoltzmannMachine<f32>, vals: &[u8]) {
    let slice = machine.values_mut();
    for v in &mut *slice {
        *v = -0.0;
    }

    for (i, &v) in vals.iter().enumerate() {
        if v > 0 && v < 10 {
            for k in 1..10 {
                if v == k {
                    slice[i*9 + k as usize - 1] = 1.0;
                } else {
                    slice[i*9 + k as usize - 1] = -0.0;
                }
            }
        }
    }
}

fn list_fixed(sudoku: &[u8]) -> Vec<usize> {
    let mut fixed = Vec::new();
    for (i, &v) in sudoku.iter().enumerate() {
        if v > 0 && v < 10 {
            for j in 0..9 {
                fixed.push(9*i + j);
            }
        }
    }
    fixed
}

fn generate_links() -> SymmetricMatrix<f32> {
    let mut matrix = SymmetricMatrix::zeros(81*9);
    for i in 0..3 {
    for j in 0..3 {
    for x in 0..3 {
    for y in 0..3 {
        let me = sudoku_to_index((i,j,x,y));
        // connect same case
        for u in 0..9 {
        for v in 0..u {
            matrix[(9*me + u, 9*me + v)] = -100.0;
        }
        }
        // connect to same square
        for u in 0..3 {
        for v in 0..3 {
            if u == x && v == y { continue; }
            let other = sudoku_to_index((i,j,u,v));
            for val in 0..9 {
                matrix[(9*me + val, 9*other + val)] = -100.0;
            }

        }}
        // connect to same column
        for u in 0..3 {
        for v in 0..3 {
            if u == i && v == x { continue; }
            let other = sudoku_to_index((u,j,v,y));
            for val in 0..9 {
                matrix[(9*me + val, 9*other + val)] = -100.0;
            }

        }}
        // connect to same row
        for u in 0..3 {
        for v in 0..3 {
            if u == j && v == y { continue; }
            let other = sudoku_to_index((i,u,x,v));
            for val in 0..9 {
                matrix[(9*me + val, 9*other + val)] = -100.0;
            }
        }}
    }}}}
    matrix
}

fn display_machine(machine: &BoltzmannMachine<f32>) {
    let vals = machine.values();
    let mut disp = vec![0u8; 81];
    for i in 0..81 {
        for j in 0..9 {
            if vals[i*9+j] > 0.5 {
                if disp[i] > 0 {
                    disp[i] = 10;
                } else {
                    disp[i] = (j+1) as u8;
                }
            }
        }
    }
    display_sudoku(&disp);
}

fn main() {
    display_sudoku(&INPUT_SUDOKU);
    let links = generate_links();
    let mut machine = BoltzmannMachine::with_biases(links, ::std::iter::repeat(10.0).take(81*9).collect());
    let fixed = list_fixed(&INPUT_SUDOKU);
    set_values(&mut machine, &INPUT_SUDOKU);
    let mut temperature = 60.0;
    let mut ticks: usize = 200;
    display_machine(&machine);
    loop {
        // read input
        print!("Temperature ? [{}] :", temperature);
        ::std::io::stdout().flush();
        let mut buf = String::new();
        let _ = ::std::io::stdin().read_line(&mut buf);
        match buf.trim().parse() {
            Ok(t) => { temperature = t; }
            Err(_) => {}
        }
        print!("Tick count ? [50] :");
        ::std::io::stdout().flush();
        buf.clear();
        let _ = ::std::io::stdin().read_line(&mut buf);
        match buf.trim().parse() {
            Ok(0) => return,
            Ok(t) => { ticks = t; },
            Err(_) => {}
        };
        for _ in 0..ticks {
            machine.tick_one_random(temperature, &fixed);
            display_machine(&machine);
            std::thread::sleep_ms(10);
        }
    }
}
