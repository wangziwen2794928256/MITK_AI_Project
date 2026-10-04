import sys

from config import MODEL_PATH



def inference(input_path,output_path):


    print(
    "Input:",
    input_path
    )


    print(
    "Model:",
    MODEL_PATH
    )


    print(
    "Output:",
    output_path
    )



    # 后续替换为nnUNet预测


    return True




if __name__=="__main__":


    input_path=sys.argv[1]


    output_path=sys.argv[2]


    inference(

    input_path,

    output_path

    )                                                                                                                                                                                  