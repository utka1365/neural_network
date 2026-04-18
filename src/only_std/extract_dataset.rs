use std::fs::File;
use std::io::Read;

// load images from MNIST dataset
pub fn mnist_input(path: String) -> Result<Vec<Vec<f32>>, std::io::Error>{
    let mut inputs: Vec<Vec<f32>> = Vec::new();

    let mut file: File = File::open(path)?;
    let mut buffer: Vec<u8> = Vec::new();

    file.read_to_end(&mut buffer)?;

    let mut i: usize = 16;
    while i < buffer.len(){
        let mut curr_input: Vec<f32> = Vec::new();
        for _ in 0..784{
            curr_input.push(buffer[i] as f32 / 255.0);
            i += 1;
        }
        inputs.push(curr_input);
    }

    Ok(inputs)
}

// load labels for images from MNIST dataset
pub fn mnist_output(path: String) -> Result<Vec<Vec<f32>>, std::io::Error>{
    let mut outputs: Vec<Vec<f32>> = Vec::new();

    let mut file: File = File::open(path)?;
    let mut buffer: Vec<u8> = Vec::new();

    file.read_to_end(&mut buffer)?;

    for i in 8..buffer.len(){
        outputs.push(vec![0.0; 10]);
        outputs[i-8][buffer[i] as usize] = 1.0;
    }

    Ok(outputs)
}
/*
pub fn xor_dataset(path: String) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), std::io::Error>{

}*/