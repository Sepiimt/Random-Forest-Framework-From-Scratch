import numpy as np
import os
from typeguard import typechecked
from typing import Self
from ..api.decorators import requires_fit

#> ---------------------------------------------------------------------------------------

class Encoder:
    # --- Documentation ---
    """\u200b
    --- Encoder Class ---
    Encoder implemented from scratch by "Sepanta Metanat"

    First edit: "2026/08/1"
    Last edit: "2026/08/5"
    """
    def __init__(self):
        self.decoder_data_path = None
        self.encoded_columns = None
        self.is_fitted: bool = False

    @typechecked
    def fit_transform(self,
                      X: np.ndarray,
                      columns: list,
                      save_path: str = r"../artifacts/encoder/model_data/"
                      ) -> np.ndarray:
        # --- Documentation ---
        """\u200b
        #> Usage and Information
        Use this function to train Encoder model while encoding certain columns.
            
        #> Parameters Documentation: 
        1. X: X array
        2. columns: Columns to get encoded
        3. save_path: Save-path for saving the model info 
        \t(to be able to encode new data based on training)
        """
        # --- Changing dtype ---
        X = X.astype(object)
        # --- Path Check ---
        os.makedirs(save_path, exist_ok=True)
        # --- Saving Data ---
        self.decoder_data_path = save_path
        self.encoded_columns = columns.copy()
        decoder_source_dict = {}
        # --- Encoding and Saving ---
        for column_num in columns:
            decoder_x, encoded_x = np.unique(X[:, column_num], 
                                             return_inverse=True)
            X[:, column_num] = encoded_x
            # --- Saving Decoder ---
            decoder_source_dict[column_num] = decoder_x
        # --- Saving Dict File ---
        np.save(os.path.join(save_path, "decode_data.npy"), 
                decoder_source_dict)
        # --- Flag as Fitted ---
        self.is_fitted = True
        # --- Return ---
        return X

    @requires_fit
    @typechecked
    def encoder(self, X: np.ndarray) -> np.ndarray:
        # --- Documentation ---
        """\u200b
        #> Usage and Information
        Use this function to encoding pre-defined columns based on training info.
            
        #> Parameters Documentation: 
        1. X: X array
        """
        # --- Changing dtype ---
        X = X.astype(object)
        # --- Loading Related Data ---
        decoder_source_dict_path = os.path.join(self.decoder_data_path, 
                                                "decode_data.npy")
        encode_source = np.load(decoder_source_dict_path, 
                                allow_pickle=True).item()
        # --- Looping ---
        for column_num in self.encoded_columns:
            vocab = encode_source[column_num]
            col_data = X[:, column_num]
            # --- Strict Unseen Data Check ---
            is_known = np.isin(col_data, vocab)
            if not np.all(is_known):
                unseen = np.unique(col_data[~is_known])
                raise ValueError(f"Column {column_num} contains unseen categories: {unseen}")
            # --- Encoding ---
            X[:, column_num] = np.searchsorted(vocab, col_data)
        # --- Return ---
        return X

    @requires_fit
    @typechecked
    def decoder(self, X: np.ndarray) -> np.ndarray:
        # --- Documentation ---
        """\u200b
        #> Usage and Information
        Use this function to decode encoded data back to original form.
        \t (based on training info)
            
        #> Parameters Documentation: 
        1. X: X array
        2. columns: Columns to get encoded
        3. save_path: Save-path for saving the model info 
        \t(to be able to encode new data based on training)
        """
        # --- Changing dtype ---
        X = X.astype(object)
        # --- Loading Related Data ---
        decoder_source_dict_path = os.path.join(self.decoder_data_path, 
                                                "decode_data.npy")
        decoder_source = np.load(decoder_source_dict_path, 
                                 allow_pickle=True).item()
        # --- Decoding ---
        for column_num in self.encoded_columns:
            vocab = decoder_source[column_num]
            indices = X[:, column_num].astype(int)
            # --- Replacing Column ---
            X[:, column_num] = vocab[indices]
        # --- Return ---
        return X

    @requires_fit
    @typechecked
    def save_model(self, 
                   save_dir_path: str = r"../artifacts/encoder/model_data/",
                   silent_save: bool = False) -> None:
        # --- Documentation ---
        """\u200b
        #> Usage and Information
        Use this function to save the trained Encoder class.
            
        #> Parameters Documentation: 
        1. save_dir_path: Save-path directory
        2. silent_save: Flag to silent the success report
        """
        # --- Path Check ---
        os.makedirs(save_dir_path, exist_ok=True)
        # --- Model Data ---
        model_data = {
            "decoder_data_path": self.decoder_data_path,
            "encoded_columns": self.encoded_columns}
        # --- Saving ---
        np.save(os.path.join(save_dir_path, "model_data.npy"), model_data)
        # --- Message ---
        if not silent_save:
            print("Model saved successfully!")


    @classmethod
    @typechecked
    def load_model(cls,
                   dir_path: str = os.path.join("..", "artifacts", 
                                                "encoder", "model_data"),
                   silent_load: bool = False) -> Self:
        # --- Documentation ---
        """\u200b
        #> Usage and Information
        Use this function to load a trained Encoder class.
            
        #> Parameters Documentation: 
        1. dir_path: Model saved-path directory
        2. silent_load: Flag to silent the success report
        """
        # --- Creating Specific Path ---
        load_path = os.path.join(dir_path, "model_data.npy")
        # --- Path Check ---
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model data found at {dir_path}!")
        # --- Reading .npy File ---
        model_data = np.load(load_path, allow_pickle=True).item()
        # --- Instancing ---
        instance = cls()
        # --- Loading Data ---
        instance.decoder_data_path = model_data["decoder_data_path"]
        instance.encoded_columns = model_data["encoded_columns"]
        # --- Message ---
        if not silent_load:
            print("Model loaded successfully!")
        # --- Return ---
        return instance
    
#> ---------------------------------------------------------------------------------------
