//i think - hmm mabe me make tensor class? FOR PyTorch? l know pytorch using nn-module for worked C++ 
//my task - connekted* nn module in my c++ tensor class, and anderstud how worked nn-module)))
//i inderstand!!! i have to - have autograd becose autograd have Softmax, CrossEntropy and all matrix operation!
//dont include this function in random position, this include in autograd

//done. test in rust, ok?

struct Tensor
{
    data:Vec<f32>,//dinamический массив(), просто данные в памяти 
    //вот так будут лежать просто вектор float32 
    //без информации кто где
    shape: Vec<usize>//размеры тензора по каждой из осей 
    //как раз shape и делит data на части
}

impl Tensor
{
    fn new(data: Vec<f32>, shape: Vec<usize>)->Tensor
    {
        Tensor
        {
            data,
            shape        
        }
    }
    fn embedding_lookup(token: usize, embedding_matrix: &Vec<Vec<f32>>) -> Tensor 
    {
    Tensor::new(embedding_matrix[token].clone(), vec![embedding_matrix[token].len()])
    }


}
