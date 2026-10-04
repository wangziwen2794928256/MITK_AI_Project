import nibabel as nib



def save_nifti(data,path):


    img=nib.Nifti1Image(

        data,

        affine=None

    )


    nib.save(

        img,

        path

    )