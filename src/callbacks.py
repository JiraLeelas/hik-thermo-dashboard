import dash
from dash import Dash, Input, Output, ALL, MATCH, Patch, State, dcc, html, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import datetime
import string
import base64
import io
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from scipy.optimize import curve_fit
import numpy as np

import app

######################################
########## Global Variables ##########
###################################### 
 
rows = list(string.ascii_uppercase[:8])# A to H
cols = list(range(1, 13)) # 1 to 12
wellplate_ids = [f"{row}{col}"
    for row in rows
    for col in cols]
time_diff_ids = [f"rate_{a_well}" for a_well in wellplate_ids]

### Dicts for dropdown
ids_dct = [
    {"label": f"{row}{col}", "value": f"{row}{col}"}
    for row in rows
    for col in cols]

rows_dct = [{"label": f"Row {row}", "value": f"{row}"} for row in rows] # Row A to H
cols_dct = [{"label": f"Column {col}", "value": f"{col}"} for col in cols] # Column 1 to 12

### Support Adv Operations
fit_operations = ["fit-sim-lin", "fit-expo-growth", "fit-expo-decay", "fit-poly", "fit-new-cool"]
fit_op_dict = {
    "fit-sim-lin": "Linear", 
    "fit-expo-growth": "Exponential Growth", 
    "fit-expo-decay": "Exponential Decay", 
    "fit-poly": "Polynomial", 
    "fit-new-cool": "Newtonian Cooling"
}
fore_operations = ["fore-smooth", "fore-mov-avg", "fore-expo"]

### Global Styling
style_table_header = {"whiteSpace": "normal", 
                        "width": "auto",
                        "padding": "10px",
                        "text-align": "center",
                        "font-weight": "bold"}

style_table_cell = {"padding": "10px", 
                        "text-align": "center"}

style_table_dash = {"height": "600px", 
                        "overflowX": "auto",
                        "overflowY": "auto"}

#####################################
############# Functions #############
#####################################
 
### Styling Function
def get_min_width(col_name):
    min_width = 42
    char_width = 8
    return min_width + (char_width * len(col_name))

def style_table(dataframe):
    ## Apply Styling based on character width
    style_data_con = list()
    style_cell_con = list()
    for col in dataframe.columns:
        min_px_width = get_min_width(col)
        style_data_con.append({
                "if": {"column_id": col},
                "minWidth": f"{min_px_width}px",
                "whiteSpace": "normal"
            })
        style_cell_con.append({
                "if": {"column_id": col},
                "minWidth": f"{min_px_width}px",
                "whiteSpace": "normal"
            })
    return style_data_con, style_cell_con

### Check Upload CSV format
def check_upload_format(filename, dataframe):
    ## Function to check dataframe columns format
    required_cols = ["Date", "Time"] + wellplate_ids ## Columns of "Date", "Time", "A1", "A2", ..., "H12"
    if not all(col in dataframe.columns for col in required_cols):
        print(f"Error: Uploaded file {filename} is missing the required columns.")
        return False
    return True

def parse_contents(contents, filename):
    df = None # Define dataframe variable

    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        try:
            if "csv" in filename:
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

            elif "xls" in filename:
                df = pd.read_excel(io.BytesIO(decoded))
        
        except:
            return html.Div([
                "Error: There was an error processing this file.",
                "Please upload an 'xls' or 'csv' file."
            ])

        if check_upload_format(filename, df):
            return {"filename": filename, "dataframe": df.to_dict()}

        else:
            print("Error: Uploaded file is in incorrect format")
            return html.Div([
                "Error: Uploaded file is in incorrect format"
            ])

    else:
        print("Error: No uploaded content")
        return html.Div([
            "Error: No Contents to be parse."
        ])

### Convert Functions
def convert_timestamp(dataframe):
    ## Function to convert 'Date' and 'Time' column into usable format.
    df = dataframe.copy()
    if "Date" and "Time" in df.columns:
        df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
        reorder_list = ["DateTime"] + [col for col in df.columns if col not in ["DateTime", "Date", "Time"]]
        new_dataframe = df[reorder_list] # create new dataframe
        return new_dataframe
    else:
        print("Error: The dataframe does not contain 'Date' and 'Time' columns.")
        return None

def convert_seconds(dataframe):
    ## Function to convert 'DateTime' column into seconds.
    df = dataframe.copy()
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        first_timestamp = df["DateTime"].iloc[0]
        df["Time"] = df["DateTime"].apply(lambda x: (x - first_timestamp).total_seconds())
        return df
    else:
        print("Error: The dataframe does not contain 'DateTime' column.")
        return None

def convert_minutes(dataframe):
    ## Function to convert 'DateTime' column into minutes.
    df = dataframe.copy()
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        first_timestamp = df["DateTime"].iloc[0]
        df["Time"] = df["DateTime"].apply(lambda x: np.round((x - first_timestamp).total_seconds()/60, 2))
        return df
    else:
        print("Error: The dataframe does not contain 'DateTime' column.")
        return None

def convert_rate_ls(well_list):
    ## Transform a list of ids into rate ids "A1" to "rate_A1"
    if isinstance(well_list, list):
        return [f"rate_{a_well}" for a_well in well_list]
    else:
        print("Error: The argument should be a list.")
        return None

### Get elements functions
def get_title(filename):
    ## This function outputs a title for a plot form the filename
    if filename.endswith((".xls", ".csv")):
        file_title, file_extension=filename.rsplit(".", 1)
        return file_title.title()
    else:
        print("Error: Invalid File Extension")
        return filename

def get_multiFileTitle(file_list, title_txt):
    ## This function creates a compare title name from files Comparision Plot of {title}, {title2}, ... and {title3}
    if not isinstance(file_list, list):
        file_list = [file_list] # Make sure that its a list
    file_list = [fname.split(".",1)[0] for fname in file_list] # remove extension

    if title_txt is None:
        title_txt = "Comparision Plot"

    if not file_list: # Empty list
        return f"{title_txt}"
    elif len(file_list) == 1:
        return f"{title_txt} of {file_list[0]}"
    elif len(file_list) == 2:
        return f"{title_txt} of {', '.join(file_list[:-1])} and {file_list[-1]}"
    else:
        return f"{title_txt} of {', '.join(file_list[:-1])}, and {file_list[-1]}"

def get_wells_str(well_list):
    ## Take in a list ["1", "2", "3"] and return a string 1, 2, 3
    return ", ".join(well_list)

def get_dataframe(filename, uploaded_lod):
    ## Function to return a specific file from the dcc.Store.
    if len(uploaded_lod) != 0 and filename != None:
        filename_list = [entry["filename"] for entry in uploaded_lod]
        index = filename_list.index(filename)
        data_dict = uploaded_lod[index]["dataframe"]
        data_df = pd.DataFrame.from_dict(data_dict)
        return data_df
    else:
        print("Error: file loading error")
        return None

def get_rate_dataframe(dataframe, time_unit="unit-sec", decimals=4):
    accept_units = ["unit-sec", "unit-min"] # Naming convention of this program
    df = dataframe.copy()
    temp_rate_list = list()
    if dataframe is not None and time_unit in accept_units:
        ## Convert into desired units
        if time_unit == "unit-sec":
            df = convert_seconds(convert_timestamp(df))
        else:
            df = convert_minutes(convert_timestamp(df))
        ## Calculate Time Difference
        df["Time_Diff"] = df["Time"].diff()

        ## Calculate Rate
        for a_well in wellplate_ids:
            temp_diff_id = f"rate_{a_well}"
            temp_diff = df[a_well].diff() # Example: get time diff of A1
            temp_change_rate = temp_diff / df["Time_Diff"]
            temp_change_rate = temp_change_rate.round(decimals)
            temp_rate_list.append(temp_change_rate.rename(temp_diff_id))

        ## Create Temp Rate Dataframe
        temp_rate_df = pd.concat([df["DateTime"], df["Time"], df["Time_Diff"]] + temp_rate_list, axis=1)
        
        return temp_rate_df
        
    else:
        print(f"Error: Please recheck input values. Accepted units are {accept_units}")
        return None

def handle_convert_df(dataframe, data_type, time_unit):
    ## This function encapsule functions to convert into specific time units
    available_types = ["temp", "rate"]
    clone_df = dataframe.copy()
    convert_df = pd.DataFrame()

    ## Error Handling
    if data_type not in available_types:
        print(f"Error: Incompatible data, this function handles {available_types} data.")
        return None

    ## Match data_type
    match data_type:
        case "temp":
            if time_unit == "unit-min":
                convert_df = convert_minutes(convert_timestamp(clone_df)) # df: DateTime. Time. Wells'
            else:
                convert_df = convert_seconds(convert_timestamp(clone_df)) # df: DateTime. Time. Wells'
        case "rate":
            convert_df = get_rate_dataframe(clone_df, time_unit)
    
    return convert_df
       

def get_TempOrRateList(data_type, select_view, select_group):
    available_types = ["temp", "rate"]
    select_cols = list()
    ## Error handling
    if data_type not in available_types:
        print(f"Error: Incompatible data, this function handles {available_types} data.")
        return None
    
    ## Match data_type
    match data_type:
        case "temp":
            if select_view == "plt-all" or select_group == None:
                select_cols = wellplate_ids
            elif select_view == "plt-rows" and select_group != None:
                for a_group in select_group:
                    select_cols = select_cols + get_rows(a_group)
            elif select_view == "plt-cols" and select_group != None:
                for a_group in select_group:
                    select_cols = select_cols + get_cols(a_group)
            elif select_view == "plt-custom" and select_group != None:
                select_cols = select_group

        case "rate":
            if select_view == "plt-all" or select_group == None:
                select_cols = time_diff_ids
            elif select_view == "plt-rows" and select_group != None:
                for a_group in select_group:
                    select_cols = select_cols + get_rate_rows(a_group)
            elif select_view == "plt-cols" and select_group != None:
                for a_group in select_group:
                    select_cols = select_cols + get_rate_cols(a_group)
            elif select_view == "plt-custom" and select_group != None:
                select_cols = convert_rate_ls(select_group)

    return select_cols

def get_spec_df_cols(dataframe, select_cols, time_unit):
    ## Check Data Structure
    rq_temp_cols = ["Date", "Time"] + wellplate_ids
    rq_rate_cols = ["Date", "Time"] + time_diff_ids
    clone_df = dataframe.copy() # Clone
    if (col in clone_df.columns for col in rq_temp_cols) or (col in clone_df.columns for col in rq_rate_cols):
        df_time_unit = "Seconds" # default

        if time_unit == "unit-min":
            clone_df["Minutes"] = clone_df["Time"]
            df_time_unit = "Minutes"
        else:
            clone_df["Seconds"] = clone_df["Time"]
        
        ## Create "Date" and "Time" column if "DateTime" exist
        if "DateTime" in clone_df.columns:
            clone_df["Date"] = pd.to_datetime(clone_df["DateTime"]).dt.date
            clone_df["Time"] = pd.to_datetime(clone_df["DateTime"]).dt.time 
            
        ## Prepare to return as select_df
        select_df = clone_df[["Date", "Time", df_time_unit] + select_cols]
        return select_df
    
    else:
        print("Error: Dataframe does not contain required columns")
        return None

def get_rows(row):
    return [f"{row}{col}" for col in cols]

def get_cols(col):
    return [f"{row}{col}" for row in rows]

def get_rate_rows(row):
    return [f"rate_{row}{col}" for col in cols]

def get_rate_cols(col):
    return [f"rate_{row}{col}" for row in rows]

### Advance Functions to merge and compare
def rename_cols_with_fname(dataframe, filename, ignore_col="Time"):
    ## Rename columns to filename + cols
    clone_df = dataframe.copy() # Clone
    new_col_names = {col: f"{filename}_{col}" if col != ignore_col else col for col in clone_df.columns}
    clone_df = clone_df.rename(columns=new_col_names)
    return clone_df

def get_temp_dfs(upload_data, fname_list, select_wells, time_unit="unit-sec"):
    ## Extract subset of dataframe and rename it
    extracted_dfs = list()
    ## Error Checking
    if not isinstance(fname_list, list):
        print("Error: provided filenames argument is not a list.")
        return None
    if not isinstance(select_wells, list):
        print("Error: provided wells argument is not a list.")
        return None
    if len(fname_list) == 0:
        print("Error: No fname provided")
        return None

    ## loop through the list
    for a_file in fname_list:
        df_name = a_file.rsplit(".", 1)[0].lower() # Cut the file extension retain lowercase filename
        ## Get dataframe and change unit
        select_df = get_dataframe(a_file, upload_data)
        if time_unit == "unit-min":
            select_df = convert_minutes(convert_timestamp(select_df))
            x_axis_lab = "Time (Minutes)"
        else:
            select_df = convert_seconds(convert_timestamp(select_df))

        ## Get Specific columns
        extracted_df = select_df[["Time"] + select_wells] # Get only specific columns
        extracted_df = rename_cols_with_fname(extracted_df, df_name) # rename to make each col unique
        # print(extracted_df)
        extracted_dfs.append(extracted_df) # get a list of dataframes

    return extracted_dfs

## Playing with timeseries
def ts_convert_unit(ts_data, time_unit="unit-sec"):
    if time_unit == "unit-min":
        return ts_data.dt.hour * 60 + ts_data.dt.minute + ts_data.dt.second / 60
    else:
        return ts_data.dt.hour * 3600 + ts_data.dt.minute * 60 + ts_data.dt.second

def to_sec_col(dataframe, time_col, time_unit):
    ## Convert a non datetime column into seconds
    clone_df = dataframe.copy()
    if time_unit == "unit-min":
        clone_df[time_col] = clone_df[time_col] * 60
    return clone_df

def merge_temp_dfs(df_list, merged_col="Time", time_unit="unit-sec", sec_tolerance=1.5):
    """
    This functions apply merge using a reference dataframe[x_col] and merge with tolerance.
    It is relatively invasive and result in loss of information.
    """
    accepted_units = ["unit-sec", "unit-min"]
    ## Error Checking
    if time_unit not in accepted_units:
        print("Error: provided argument unit is not accepted.")
        return None
    if not isinstance(df_list, list):
        print("Error: provided argument is not a list.")
        return None
    
    ## Convert into seconds columns before merging
    df_list = [to_sec_col(df, merged_col, time_unit) for df in df_list]
    for df in df_list:
        df.sort_values(by=merged_col, inplace=True)
    
    merged_df = df_list[0] # Merged from first df

    for a_df in df_list[1:]:
        merged_df = pd.merge_asof(
            merged_df,
            a_df,
            on=merged_col,
            direction="nearest",
            tolerance=sec_tolerance
        )
    
    if time_unit == "unit-min":
        merged_df[merged_col] = merged_df[merged_col] / 60
    
    return merged_df

def ts_bounded_interpolate(df, time_col, bound_tuple, method="linear"):
    ## Error Checking
    if isinstance(bound_tuple, tuple) and len(bound_tuple) != 2:
        print("Error: the provided boundary tuple is incompatible.")
        return None
    clone_df = df.copy()
    min_bound, max_bound = bound_tuple[0], bound_tuple[1] # Get boundaries

    clone_df = clone_df.set_index(time_col) # Set index for interpolation
    clone_df = clone_df.interpolate(method=method)
    clone_df = clone_df.reset_index()

    ## Apply Contrain
    clone_df.loc[clone_df[time_col] < min_bound, clone_df.columns.difference([time_col])] = None
    clone_df.loc[clone_df[time_col] > max_bound, clone_df.columns.difference([time_col])] = None

    return clone_df

def combine_temp_dfs(df_list, time_col="Time", time_unit="unit-sec"):
    """
    This function creates a unified time index from all the dataframes then join them together.
    If the time gap does not exist, it will leave as NaN.
    """
    ## Error Checking
    accepted_units = ["unit-sec", "unit-min"]
    if time_unit not in accepted_units:
        print("Error: provided argument unit is not accepted.")
        return None
    if not isinstance(df_list, list):
        print("Error: provided argument is not a list.")
        return None
    
    combined_df = None

    if len(df_list) == 1:
        combined_df = df_list[0].copy() # get the first one

    else:
        df_list = [to_sec_col(df, time_col, time_unit) for df in df_list] # Convert to total seconds
        combined_time_index = sorted(pd.concat([df[time_col] for df in df_list]).unique()) # get pd series of time
        reindexed_dfs = []
        for df in df_list:
            reindexed_df = df.set_index(time_col).reindex(combined_time_index).reset_index()
            print(reindexed_df)
            reindexed_dfs.append(reindexed_df)
        
        ## Combined
        combined_df = reindexed_dfs[0] # get first dataframe
        for df in reindexed_dfs[1:]:
            combined_df = pd.merge(combined_df, df, on=time_col, how="outer")
        
        if time_unit == "unit-min": # Only combined idx as we convert them all to seconds
            combined_df[time_col] = combined_df[time_col] / 60
    
    return combined_df

### Advance Curvefitting
def obj_linear(t, m, c):
    """
    Equation:
        T(t) = mt+c
    
    Parameters:
        T(t) = Temperature
        t = Time
        m = slope
        c = y-intercept
    """
    return (m*t) + c

def obj_expo_growth(t, T_zero, k):
    """
    Equation:
        T(t) = T_zero*e^(kt)
        
    Parameters:
        T_zero = Temperature of object at time equals 0.
        k = constant
        t = time
    """
    return T_zero*np.exp(k*t)

def obj_expo_decay(t, T_zero, k):
    """
    Equation:
        T(t) = T_zero*e^(-kt)
        
    Parameters:
        T_zero = Temperature of object at time equals 0.
        k = constant
        t = time
    """
    return T_zero*np.exp(-1*k*t)

def obj_polynomial(x, *coeffs):
    """
    Equation:
        y = a0 + a1*x + a2*x^2 + ... + an*x^n

    Parameters:
        x = input values
        *coeffs = argument list corresponds to the constant term of polynomial functions
    """
    return sum(c * x**i for i, c in enumerate(coeffs))

def obj_newtonian_cooling(t, T_amb, T_zero, k):
    """
    Equation: 
        T(t) = T_amb + (T_zero - T_amb)e^(-kt)
        
    Parameters:
        T_amb = Ambient Temperature
        T_zero = Temperature of object at time equals 0.
        k = constant
        t = time
    """
    return T_amb + (T_zero - T_amb)*np.exp(-k*t)

def fit_temp_data(dataframe, x_col, y_col, select_obj_func, poly_deg=2, T_env=20):
    """
    This function fits the provided data into selected objective functions.
    It is recommended to scaled the 'x_col' in this case is time, into minutes, as to prevent overflow issue.
    """
    ## Extract df columns
    x = dataframe[x_col]
    y = dataframe[y_col]
    T_zero = y.iloc[0]

    ## Prevent None input from unsupport numbers
    poly_deg = 2 if poly_deg is None else poly_deg
    T_env = 20 if T_env is None else T_env

    ## Fit x and y data with the objective function
    support_func = {
        "fit-sim-lin": (obj_linear, (1, 1)),
        "fit-expo-growth": (obj_expo_growth, (T_zero, 0.1)),
        "fit-expo-decay": (obj_expo_decay, (T_zero, 0.1)),
        "fit-poly": (obj_polynomial, [1] * (poly_deg + 1)),
        "fit-new-cool": (obj_newtonian_cooling, (T_env, T_zero, 0.1))
    }
    if select_obj_func not in support_func.keys():
        print("Error: unsupported objective function")
        return None, None

    func, initial_guess = support_func[select_obj_func] # Get the function to be used
    params, params_cov = curve_fit(func, x, y, p0=initial_guess) # Apply curve fitting

    return func, params

def get_params_desc(select_obj_func, params):
    ## This function outputs an equation of the fitted data
    match select_obj_func:
        case "fit-sim-lin": # y = mx + c
            return f"T(t) = {params[0]:.2f}*x + {params[1]:.2f}"
        case "fit-expo-growth": # T(x) = T_zero*e^(kx)0
            return f"T(t) = {params[0]:.2f}*e^({params[1]:.4f}*t)"
        case "fit-expo-decay": # T(t) = T_zero*e^(-kt)
            return f"T(t) = {params[0]:.2f}*e^(-{params[1]:.4f}*t)"
        case "fit-poly": # y = a0 + a1*x + a2*x^2 + ... + an*x^n
            return f"T(t) = {' + '.join([f'{coef:.3f}*x^{i}' for i, coef in enumerate(params)])}"
        case "fit-new-cool": # T(t) = T_amb + (T_zero - T_amb)e^(-kt)
            return f"T(t) = {params[0]:.2f} + ({params[1]:.2f} - {params[0]:.2f})*e^(-{params[2]:.2f}*t)"
        case _:
            print("Error: Unsupport objective funciton.")
            return None

def evaluate_curve_fit(dataframe, x_col, y_col, func, params):
    ## Extract from dataframe
    x = dataframe[x_col]
    y_act = dataframe[y_col] # act = actual data

    ## Calculate Evaluation Metrics
    y_pred = func(x, *params)
    residuals = y_act - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_act - np.mean(y_act))**2)

    ## Evaluation Metrics
    r_squared = 1 - (ss_res / ss_tot)
    RMSE = np.sqrt(np.mean(residuals**2))
    MAE = np.mean(np.abs(residuals))
    
    return r_squared, RMSE, MAE
    
def create_eval_df(dataframe, x_col, y_cols, select_obj_func, poly_deg=2, T_env=20):
    ## This function structures fits and turn fit information into dataframe
    results = list()
    for y_col in y_cols:
        func, params = fit_temp_data(dataframe, x_col, y_col, select_obj_func, poly_deg, T_env)
        func_desc = get_params_desc(select_obj_func, params)
        r_squared, RMSE, MAE = evaluate_curve_fit(dataframe, x_col, y_col, func, params)
        col_result = {
            "Well": y_col,
            "Fit Type": fit_op_dict[select_obj_func],
            "Description": func_desc,
            "Parameters": ", ".join([f"{para:.5f}" for para in params]), 
            "R-squared": f"{r_squared:.3f}",
            "RMSE": f"{RMSE:.3f}",
            "MAE": f"{MAE:.3f}"
        }
        results.append(col_result)
    result_df = pd.DataFrame(results)
    return result_df

#####################################
############# Callbacks ############# 
#####################################

@callback(Output("upload-file-dropdown", "options"),
            Output("adv-file-dropdown", "options"),
            Output("upload-data-storage", "data"),
            Input("upload-data", "contents"),
            State("upload-data", "filename"))
def update_dropdown(contents_list, filenames_list):
    ## Function to update dropdown based on the uploaded data
    options = [{"label": "", "value": ""}]
    if contents_list is not None:
        uploaded_data = [parse_contents(c, n) for c, n in zip(contents_list, filenames_list)] # Reset with new content
        options = [entry["filename"] for entry in uploaded_data]
    else:
        uploaded_data = None
        options = [{"label": "", "value": ""}]
    adv_options = options
    print("Uploaded Files:", options)
    # print(uploaded_data)

    return options, adv_options, uploaded_data

@callback(Output("upload-status-txt", "children"),
            Input("upload-data-storage", "data"))
def update_upload_text(upload_data):
    ## Updates the upload file text from "No files uploaded" to "Uploaded 2 files"
    status_text = "No files uploaded"
    if upload_data is not None:
        n_files_uploaded = len(upload_data) # It is a list of dicts
        plural_s = ""
        if n_files_uploaded != 1:
            plural_s = "s"
        status_text = f"Uploaded {n_files_uploaded} file{plural_s}"

    return status_text

@callback(Output("select-row-column", "disabled"),
            Input("plt-option", "value"))
def disable_group_selection(plot_option):
    ## Disable select row or column
    if plot_option == "plt-all":
        return True
    return False

@callback([Output("tab-data-table", "disabled"),
            Output("tab-rate-table", "disabled"),
            Output("select-download-data", "disabled")], 
            Input("upload-file-dropdown", "value"))
def disable_table_tabs(select_file):
    ## Disable clickable objects when no data is selected
    if select_file is not None:
        return False, False, False
    return True, True, True

@callback(Output("select-row-column","options"),
            Output("select-row-column", "placeholder"),
            Input("plt-option", "value"))
def update_selection_options(plot_option):
    ## Update the Row / Column selection options and placeholder text
    placeholder_text = "Select..."
    options = [{"label": "", "value": ""}]
    if plot_option == "plt-rows":
        placeholder_text = "Select a row"
        options = rows_dct
    elif plot_option == "plt-cols":
        placeholder_text = "Select a column"
        options = cols_dct
    elif plot_option == "plt-custom":
        placeholder_text = "Select a well"
        options = ids_dct

    return options, placeholder_text

@callback(Output("multi-plot-area", "children"),
            [State("upload-data-storage", "data"),
            Input("multi-tabs", "active_tab"),
            Input("upload-file-dropdown", "value"),
            Input("plt-option", "value"),
            Input("select-row-column", "value"),
            Input("select-time-unit", "value")]) 
def plot_multi_graphs(upload_data, select_tab, select_file, select_view, select_group, time_unit):
    print(f"Select Group: {select_group}")
    ## Function to display the graphs
    fig = go.Figure()
    tab_content = dcc.Graph(figure = fig)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0), # Remove all the margins
        height=500,
        autosize=True, # Auto plot size
    )
    if (upload_data and select_file) is not None:

        ### Plot Time Temp Plot
        if select_tab == "tab-temp-time":
            select_df = get_dataframe(select_file, upload_data) # Get dataframe
            plot_title = f"{get_title(select_file)}: Temperature vs. Time All Wells"
            display_hover = ["Time: %{x} Seconds","Temperature: %{y} °C"]
            x_axis_lab = "Time (Seconds)"

            ## Convert Units
            if time_unit == "unit-min": 
                select_df = convert_minutes(convert_timestamp(select_df))
                display_hover = ["Time: %{x} Minutes","Temperature: %{y} °C"]
                x_axis_lab = "Time (Minutes)"
            else:
                select_df = convert_seconds(convert_timestamp(select_df))

            ## Plot Variables
            display_df_cols = list()
            plot_title = None

            ## Plot the Graph
            if select_view == "plt-all":
                plot_title = f"{get_title(select_file)}: Temperature vs. Time All Wells"
                display_df_cols = wellplate_ids

            elif select_view == "plt-rows" and select_group != None:
                plot_title = f"{get_title(select_file)}: Temperature vs. Time for Row {get_wells_str(select_group)}"
                for a_group in select_group:
                    display_df_cols = display_df_cols + get_rows(a_group) # Get a list of rows

            elif select_view == "plt-cols" and select_group != None:
                plot_title = f"{get_title(select_file)}: Temperature vs. Time for Column {get_wells_str(select_group)}"
                for a_group in select_group:
                    display_df_cols = display_df_cols + get_cols(a_group) # Get a list of cols
            
            elif select_view == "plt-custom" and select_group != None:
                plot_title = f"{get_title(select_file)}: Temperature vs. Time for Wells {get_wells_str(select_group)}"
                display_df_cols = select_group

            ## Loop to plot
            for a_well in display_df_cols:
                fig.add_trace(go.Scatter(x=select_df["Time"], y=select_df[a_well], mode="lines", name=a_well, 
                    hovertemplate="<br>".join(display_hover)))
                fig.update_layout(title=plot_title, xaxis_title=x_axis_lab, yaxis_title="Temperature (°C)") 

        ### Plot Rate of Change
        elif select_tab == "tab-rate-time": 
            select_df = get_dataframe(select_file, upload_data) # Get dataframe
            display_hover = ["Time: %{x} Seconds", "Temperature Change Rate: %{y} °C/s"]
            x_axis_lab = "Time (Seconds)"
            y_axis_lab = "Temperature Change Rate (°C/s)"

            ## Convert Units and Get Rate Dataframe
            if time_unit == "unit-min": 
                select_df = get_rate_dataframe(select_df, time_unit)
                display_hover = ["Time: %{x} Minutes", "Temperature Change Rate: %{y} °C/min"]
                x_axis_lab = "Time (Minutes)"
                y_axis_lab = "Temperature Change Rate (°C/min)"

            else:
                select_df = get_rate_dataframe(select_df, time_unit)

            ## Plot Variables
            display_df_cols = list()
            plot_title = None
            
            ## Plot the Graph
            if select_view == "plt-all":
                plot_title = f"{get_title(select_file)}: Rate of Temperature Change All Wells"
                display_df_cols = time_diff_ids

            elif select_view == "plt-rows" and select_group != None:
                plot_title = f"{get_title(select_file)}: Rate of Temperature Change for Row {get_wells_str(select_group)}"
                for a_group in select_group:
                    display_df_cols = display_df_cols + get_rate_rows(a_group) # Get a list of rows

            elif select_view == "plt-cols" and select_group != None:
                plot_title = f"{get_title(select_file)}: Rate of Temperature Change for Column {get_wells_str(select_group)}"
                for a_group in select_group:
                    display_df_cols = display_df_cols + get_rate_cols(a_group) # Get a list of cols
            
            elif select_view == "plt-custom" and select_group != None:
                plot_title = f"{get_title(select_file)}: Rate of Temperature Change for Wells {get_wells_str(select_group)}"
                display_df_cols = convert_rate_ls(select_group)

            ## Loop to plot
            for a_well in display_df_cols:
                fig.add_trace(go.Scatter(x=select_df["Time"], y=select_df[a_well], mode="lines", name=a_well, 
                    hovertemplate="<br>".join(display_hover)))
                fig.update_layout(title=plot_title, xaxis_title=x_axis_lab, yaxis_title=y_axis_lab) 
        
        ### Display Uploaded Data As a Table
        elif select_tab == "tab-data-table":
            select_df = get_dataframe(select_file, upload_data) # Get dataframe
            display_df_cols = list() 

            ## Convert Units into desired dataframe
            select_df = handle_convert_df(select_df, "temp", time_unit)

            ## Filter Data
            display_df_cols = get_TempOrRateList("temp", select_view, select_group)

            ## Get Specific Dataframe Columns
            table_df = get_spec_df_cols(select_df, display_df_cols, time_unit)
            table_data = table_df.to_dict("records")
            table_columns = [{"name": i, "id": i} for i in table_df.columns]

            ## Apply Styling
            # style_data_con, style_cell_con = style_time_unit(table_df)
            style_data_con, style_cell_con = style_table(table_df)

            ## Tab content
            tab_content = dash_table.DataTable(
                data=table_data,
                columns=table_columns,
                fixed_rows={"headers":True},
                style_header=style_table_header,
                style_cell=style_table_cell,
                style_table=style_table_dash,
                style_data_conditional=style_data_con,
                style_cell_conditional=style_cell_con
            )
        
        ### Display Rate Table
        elif select_tab == "tab-rate-table":
            select_df = get_dataframe(select_file, upload_data) # Get dataframe
            display_df_cols = list()

            ## Convert Units and Get Rate Dataframe
            select_df = handle_convert_df(select_df, "rate", time_unit)

            ## Filter Data
            display_df_cols = get_TempOrRateList("rate", select_view, select_group)

            ## Get Specific Dataframe Columns
            table_df = get_spec_df_cols(select_df, display_df_cols, time_unit)
            table_data = table_df.to_dict("records")
            table_columns = [{"name": i, "id": i} for i in table_df.columns]

            ## Apply Styling
            # style_data_con, style_cell_con = style_time_unit(table_df)
            style_data_con, style_cell_con = style_table(table_df)

            ## Tab content
            tab_content = dash_table.DataTable(
                data=table_data,
                columns=table_columns,
                fixed_rows={"headers":True},
                style_header=style_table_header,
                style_cell=style_table_cell,
                style_table=style_table_dash,
                style_data_conditional=style_data_con,
                style_cell_conditional=style_cell_con
            )

    return tab_content 

# ## Function to disable download button

## Download Button
@callback(Output("download-multi", "data"),
            [Input("btn-download-multi", "n_clicks"), # Call this function
            State("upload-data-storage", "data"), # Data
            State("upload-file-dropdown", "value"), # Select from Data
            State("plt-option", "value"),
            State("select-row-column", "value"),
            State("select-time-unit", "value"),
            State("select-download-data", "value")],
            prevent_initial_call=True)
def download_multi_data(n_clicks, upload_data, select_file, select_view, select_group, time_unit, select_data_type):
    ## Download data based on the current selection
    custom_options = ["plt-rows", "plt-cols", "plt-custom"]
    print(select_file, select_view, select_group, time_unit, select_data_type)
    if (upload_data and select_file) is not None:
        ## Get the selected file
        select_df = get_dataframe(select_file, upload_data)
        output_df_cols = list()
        download_df = pd.DataFrame()
        file_name = select_file.rsplit(".", 1)[0] # filename without extension
        info_type = ""

        ## Select Download Type
        if select_data_type == "select-full-raw" or (select_group == None and select_data_type == "select-current-temp"): # Support the case where user does not select any loc
            output_df_cols = wellplate_ids
            select_df = handle_convert_df(select_df, "temp", time_unit)
            info_type = "original_"
        elif select_data_type == "select-full-rate" or (select_group == None and select_data_type == "select-current-rate"): # Support the case where user does not select any loc
            output_df_cols = time_diff_ids
            select_df = handle_convert_df(select_df, "rate", time_unit)
            info_type = "rate_"
        elif select_data_type == "select-current-temp" and (select_view in custom_options) and (select_group is not None):
            output_df_cols = get_TempOrRateList("temp", select_view, select_group)
            select_df = handle_convert_df(select_df, "temp", time_unit)
            info_type = "custom_temp_"
        elif select_data_type == "select-current-rate" and (select_view in custom_options) and (select_group is not None):
            output_df_cols = get_TempOrRateList("rate", select_view, select_group)
            select_df = handle_convert_df(select_df, "rate", time_unit)
            info_type = "custom_rate_"

        download_df = get_spec_df_cols(select_df, output_df_cols, time_unit)
        return dcc.send_data_frame(download_df.to_csv, f"{info_type}{file_name}.csv")
        # print("Download Success")
    else:
        print("Error: An Error occured.")

### Functions for advance plot
@callback([Output("adv-file-dropdown", "multi"), # Disable Dropdown
            Output("adv-file-dropdown", "value"), # Reset Data
            Output("adv-tabs", "active_tab"), # Change the tab
            Output("adv-operation", "disabled"), # Disable ADV Options
            Output("tab-ana", "disabled"), # Disable Ana if compare
            Output("tab-adv-plt", "disabled"), # Disable plt if ana
            Output("tab-ana-res", "disabled"),
            Output("hidden-container-parent", "style"), # If anything goes disable the whole thing
            Output("adv-time-unit", "value"), # Lock Adv time unit to minutes and prevent change -> prevent overflow
            Output("adv-time-unit", "disabled")
            ], 
            [Input("adv-mode", "value"),
            State("adv-file-dropdown", "value")])
def disable_adv_functions(select_adv_mode, select_file):
    if select_adv_mode == "adv-compare": 
        ## Active: Plot Area, Plot Data
        ## Disable: Plot Analysis, Analysis Results
        return True, None, "tab-adv-plot", True, True, False, True, {"display": "none"}, "unit-sec", False
    elif select_adv_mode == "adv-analysis": 
        ## Active: Plot Analysis, Plot Data, Analysis Results
        ## Disable: Multiple DD, Plot Area
        return False, None, "tab-adv-analysis", False, False, True, False, {"display": "block"}, "unit-min", True
    ## Default if None
    return True, None, "tab-adv-plot", True, True, False, True, {"display": "none"}, "unit-sec", False

@callback([Output("container-poly-deg", "style"),
            Output("container-room-temp", "style")],
            [Input("adv-operation", "value")])
def reveal_hidden_adv_input(select_adv_analysis):
    ## Reveal Hidden Divs
    match select_adv_analysis:
        case "fit-poly": # Get poly-deg
            return [{"display": "block"}, {"display": "none"}]
        case "fit-new-cool":
            return [{"display": "none"}, {"display": "block"}]
        case _:
            return [{"display": "none"}, {"display": "none"}]

@callback(Output("tab-adv-data", "disabled"),
            Input("adv-file-dropdown", "value"))
def disable_adv_table_tab(select_file):
    ## Disable tab if there is no selected file
    if select_file is not None:
        return False
    return True

@callback([Output("custom-plot-area", "children")],
            [State("upload-data-storage", "data"),
            Input("adv-tabs", "active_tab"),
            Input("adv-file-dropdown", "value"), # list
            Input("adv-well-select", "value"), # list
            Input("adv-time-unit", "value"), # value
            Input("adv-operation", "value"),
            Input("poly-deg", "value"), # Optional
            Input("room-temp", "value") # Optional
            ])
def plot_custom_graphs(upload_data, active_tab, select_files, select_wells, time_unit, adv_operation, poly_deg, room_temp):
    ## Plot a graph to compare from multiple dataframe
    print("Adv plot:", select_files)
    fig=go.Figure()
    adv_tab_content = dcc.Graph(figure = fig)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        autosize=True
    )
    ## Error Checking
    if (upload_data and select_files and select_wells) is None:
        return [adv_tab_content] # Early return

    ## Ensure consistant data format as a list
    if not isinstance(select_files, list):
        select_files = [select_files]
    if not isinstance(select_wells, list):
        select_wells = [select_wells]
    
    ## Match Select Tab
    match active_tab:
        case "tab-adv-plot":
            ## Variables
            x_axis_lab = "Time (Seconds)" if time_unit == "unit-sec" else "Time (Minutes)"
            y_axis_lab = "Temperature (°C)"
            plot_title = get_multiFileTitle(select_files, title_txt="Comparision Plot")

            ## Merged Multiple dataframes
            merged_df = None # Placeholder
            if select_files is not None:
                temp_df_list = get_temp_dfs(upload_data, select_files, select_wells, time_unit=time_unit)
                # merged_df = merge_temp_dfs(temp_df_list, sec_tolerance=None) ## Works but the time will cut off
                merged_df = combine_temp_dfs(temp_df_list,"Time",time_unit)
            
            # print(merged_df) # debug
            ## Loop to plot
            if merged_df is not None:
                plot_cols = [col for col in merged_df.columns if col != "Time"]
                for a_col in plot_cols:
                    fig.add_trace(go.Scatter(x=merged_df["Time"], y=merged_df[a_col], 
                        mode="lines", name=f"{a_col}",hovertemplate=None, connectgaps=True)) # Ensure that NAN are plot
                fig.update_layout(title=plot_title, xaxis_title=x_axis_lab, yaxis_title=y_axis_lab,hovermode="x unified")
            
        case "tab-adv-plot-data":
            if select_files is not None:
                temp_df_list = get_temp_dfs(upload_data, select_files, select_wells, time_unit=time_unit)
                merged_df = combine_temp_dfs(temp_df_list,"Time",time_unit)

                ## Transform into formatted data
                table_data = merged_df.to_dict("records")
                table_columns = [{"name": i, "id": i} for i in merged_df.columns]
                style_data_con, style_cell_con = style_table(merged_df)

                adv_tab_content = dash_table.DataTable(
                    data=table_data,
                    columns=table_columns,
                    fixed_rows={"headers":True},
                    style_header=style_table_header,
                    style_cell=style_table_cell,
                    style_table=style_table_dash,
                    style_data_conditional=style_data_con,
                    style_cell_conditional=style_cell_con
                )

        case "tab-adv-analysis": # Supports Single Df
            ## Variables
            x_axis_lab = "Time (Seconds)" if time_unit == "unit-sec" else "Time (Minutes)"
            y_axis_lab = "Temperature (°C)"
            plot_title = get_multiFileTitle(select_files, "Graph Fitting")
            
            if select_files is not None: 
                temp_df_list = get_temp_dfs(upload_data, select_files, select_wells, time_unit=time_unit)
                merged_df = combine_temp_dfs(temp_df_list,"Time",time_unit) ## Potential Update to Support Multi Files

                ## Get Details for Fit / Forecast
                x_col = "Time"
                y_cols = [col for col in merged_df.columns if col != x_col]
                fit_results = list()

                ## Get Size
                nrows, ncols = merged_df.shape

                ## Assign Functions
                if adv_operation in fit_operations:
                    ## loop to get data
                    for y_col in y_cols:
                        func, params = fit_temp_data(merged_df, x_col, y_col, adv_operation, poly_deg, room_temp)
                        fit_results.append((y_col, func, params))

                    ## loop to plot data
                    for y_col, func, params in fit_results:
                        fit_desc = get_params_desc(adv_operation, params)

                        ## Original Data
                        custom_data_og = np.array([y_col] * nrows)
                        fig.add_trace(go.Scatter(x=merged_df[x_col], y=merged_df[y_col], mode="lines", name=f"{y_col}",
                            customdata=custom_data_og,
                            hovertemplate=
                                "<b>Data</b>: %{customdata}"+
                                "<br><b>Time</b>: %{x:.2f}"+
                                "<br><b>Temperature</b>: %{y:.2f}", 
                            connectgaps=True))

                        ## Fitted Data
                        x_fit = np.linspace(merged_df[x_col].min(), merged_df[x_col].max(), 100)
                        y_fit = func(x_fit, *params)
                        custom_data = np.array([[y_col, fit_desc]] * len(x_fit))
                        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name=f"Fitted {y_col}",
                            customdata=custom_data,
                            hovertemplate=
                                "<b>Data</b>: %{customdata[0]}"+
                                "<br><b>Equation</b>: %{customdata[1]}"+
                                "<br><b>Time</b>: %{x:.2f}"+
                                "<br><b>Temperature</b>: %{y:.2f}", 
                            connectgaps=True))
                    
                    ## Update Plot
                    fig.update_layout(title=plot_title, xaxis_title=x_axis_lab, yaxis_title=y_axis_lab)

                elif adv_operation in fore_operations: ## Forecast Operations
                    print("Fore Operations")

                    ## No Seasonal Component Assumption

                else: ## Defalut Plot
                    for y_col in y_cols:
                        custom_data_og = np.array([y_col] * nrows)
                        fig.add_trace(go.Scatter(x=merged_df[x_col], y=merged_df[y_col], mode="lines", name=f"{y_col}",
                            customdata=custom_data_og,
                            hovertemplate=
                                "<b>Data</b>: %{customdata}"+
                                "<br><b>Time</b>: %{x:.2f}"+
                                "<br><b>Temperature</b>: %{y:.2f}", 
                            connectgaps=True))
                    
                    ## Update Plot
                    fig.update_layout(title=plot_title, xaxis_title=x_axis_lab, yaxis_title=y_axis_lab)
        
        case "tab-adv-ana-res":
            if select_files is not None:
                temp_df_list = get_temp_dfs(upload_data, select_files, select_wells, time_unit=time_unit)
                merged_df = combine_temp_dfs(temp_df_list,"Time",time_unit) ## Potential Update to Support Multi Files

                ## Get Details for Fit / Forecast
                x_col = "Time"
                y_cols = [col for col in merged_df.columns if col != x_col]
                # fit_results = list()

                ## Only Active if its in the list
                if adv_operation in fit_operations:
                    fit_result_df = create_eval_df(merged_df, x_col, y_cols, adv_operation, poly_deg, room_temp)
            
                    table_data = fit_result_df.to_dict("records")
                    table_columns = [{"name": i, "id": i} for i in fit_result_df.columns]
                    style_data_con, style_cell_con = style_table(fit_result_df)

                    adv_tab_content = dash_table.DataTable(
                        data=table_data,
                        columns=table_columns,
                        fixed_rows={"headers":True},
                        style_header=style_table_header,
                        style_cell=style_table_cell,
                        style_table=style_table_dash,
                        style_data_conditional=style_data_con,
                        style_cell_conditional=style_cell_con
                    )

                    # print(fit_result_df)

    return [adv_tab_content]

## Function To Download based selected
@callback(Output("download-adv", "data"),
            [Input("btn-download-adv", "n_clicks"), # Activator
            State("upload-data-storage", "data"), # Files
            State("adv-file-dropdown", "value"), # Names
            State("adv-well-select", "value"), # Wells
            State("adv-mode", "value"), # Data Mode
            State("select-adv-dl-data", "value"),
            State("adv-time-unit", "value"), # Min, Sec
            State("adv-operation", "value"), # Operation from a list
            State("poly-deg", "value"),
            State("room-temp", "value")],
            prevent_initial_call=True
)
def download_adv_data(n_clicks, upload_data, select_files, select_wells, select_mode, select_dl, time_unit, adv_operation, poly_deg, room_temp):
    ## output data
    adv_dl_data = None

    ## Do nothing if no essential elements are selected
    if (upload_data and select_files and select_wells) is not None:

        ## Ensure consistant data format as a list
        if not isinstance(select_files, list):
            select_files = [select_files]
        if not isinstance(select_wells, list):
            select_wells = [select_wells]

        ## Get the data
        temp_df_list = get_temp_dfs(upload_data, select_files, select_wells, time_unit)
        merged_df = combine_temp_dfs(temp_df_list, "Time", time_unit)

        ## Operation to get the filename
        file_name = "adv_operation_data"
        adv_info = ""
        if len(select_files) == 1:
            file_name = select_files[0].rsplit(".", 1)[0] # Get filename no extension
        else:
            file_name = "merged_data"

        ## Match Download Operation
        match select_dl:
            case "select-plt-data": ## DL plot data
                adv_dl_data = dcc.send_data_frame(merged_df.to_csv, f"{adv_info}{file_name}.csv")

            case "select-ana-res": ## DL analysis data
                adv_info = "Fit_"
                x_col = "Time"
                y_cols = [col for col in merged_df.columns if col != x_col]
                    
                if adv_operation in fit_operations:
                    fit_result_df = create_eval_df(merged_df, x_col, y_cols, adv_operation, poly_deg, room_temp)
                    adv_dl_data = dcc.send_data_frame(fit_result_df.to_csv, f"{adv_info}{file_name}.csv")

    return adv_dl_data