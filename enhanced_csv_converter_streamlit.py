"""

CSV Object-to-Integer Converter - Streamlit Web App

==================================================

A modern web application for converting object columns to integers

in CSV files for machine learning model training.

Features:

- Drag & drop file upload

- Interactive data preview and comparison

- Multiple encoding methods

- Real-time conversion progress

- Downloadable processed files

- Mobile-friendly responsive design

- Comprehensive null value handling

- Train/test split option

Usage:

streamlit run csv_converter_streamlit.py

Requirements: streamlit, pandas, scikit-learn

"""

import streamlit as st

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import io

import warnings

warnings.filterwarnings('ignore')

# Page configuration

st.set_page_config(

    page_title="CSV Object-to-Integer Converter",

    page_icon="🔄",

    layout="wide",

    initial_sidebar_state="expanded"

)


# Custom CSS for better styling

st.markdown("""

<style>

    .main-header {

        font-size: 2.5rem;

        font-weight: 700;

        color: #1f77b4;

        text-align: center;

        margin-bottom: 1rem;

    }



    .sub-header {

        font-size: 1.2rem;

        color: #666;

        text-align: center;

        margin-bottom: 2rem;

    }



    .success-message {

        color: #28a745;

        font-weight: bold;

    }



    .error-message {

        color: #dc3545;

        font-weight: bold;

    }



    .stProgress .st-bo {

        background-color: #1f77b4;

    }

</style>

""", unsafe_allow_html=True)

class StreamlitCSVConverter:

    """Main class for Streamlit CSV converter"""

    def __init__(self):

        self.original_df = None

        self.processed_df = None

        self.conversion_log = {}

        self.train_df = None

        self.test_df = None

        self.null_analysis = {}

    def process_uploaded_file(self, uploaded_file):

        """Process uploaded CSV file"""

        try:

            # Read CSV from uploaded file

            self.original_df = pd.read_csv(uploaded_file)

            return True, None

            # If file not read, throw an exception and return False

        except Exception as e:

            return False, str(e)

    def analyze_null_values(self):

        """Analyze null values in the dataset"""

        if self.original_df is None:

            return {}



        null_analysis = {}

        for col in self.original_df.columns:

            null_count = self.original_df[col].isnull().sum()

            null_percentage = (null_count / len(self.original_df)) * 100

            total_values = len(self.original_df)



            null_analysis[col] = {

                'null_count': null_count,

                'null_percentage': null_percentage,

                'total_values': total_values,

                'data_type': str(self.original_df[col].dtype)

            }



        self.null_analysis = null_analysis

        return null_analysis

    def handle_null_values(self, strategy_dict):

        """Handle null values based on user-selected strategies"""

        if self.original_df is None:

            return False, "No data loaded"



        self.processed_df = self.original_df.copy()



        for col, strategy in strategy_dict.items():

            if col not in self.processed_df.columns:

                continue



            null_mask = self.processed_df[col].isnull()



            if not null_mask.any():

                continue



            if strategy == 'drop_rows':

                # Drop rows with null values in this column

                self.processed_df = self.processed_df.dropna(subset=[col])



            elif strategy == 'fill_mean':

                # Fill with mean (for numeric columns)

                if pd.api.types.is_numeric_dtype(self.processed_df[col]):

                    mean_val = self.processed_df[col].mean()

                    self.processed_df[col].fillna(mean_val, inplace=True)



            elif strategy == 'fill_median':

                # Fill with median (for numeric columns)

                if pd.api.types.is_numeric_dtype(self.processed_df[col]):

                    median_val = self.processed_df[col].median()

                    self.processed_df[col].fillna(median_val, inplace=True)



            elif strategy == 'fill_mode':

                # Fill with mode (most frequent value)

                mode_val = self.processed_df[col].mode()

                if not mode_val.empty:

                    self.processed_df[col].fillna(mode_val[0], inplace=True)



            elif strategy == 'fill_forward':

                # Forward fill

                self.processed_df[col].fillna(method='ffill', inplace=True)



            elif strategy == 'fill_backward':

                # Backward fill

                self.processed_df[col].fillna(method='bfill', inplace=True)



            elif strategy == 'fill_constant':

                # Fill with a constant value (you can modify this)

                if pd.api.types.is_numeric_dtype(self.processed_df[col]):

                    self.processed_df[col].fillna(0, inplace=True)

                else:

                    self.processed_df[col].fillna('Unknown', inplace=True)



        return True, "Null values handled successfully"

    def get_object_columns(self):

        """Get list of object columns"""

        if self.processed_df is None:

            return []

        return self.processed_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    def convert_objects_to_integers(self, method='label_encoding', progress_bar=None, missing_sentinel=-1):

        """Convert object columns to integers"""

        if self.processed_df is None:

            return False, "No data loaded"



        object_cols = self.get_object_columns()



        if not object_cols:

            return True, "No object columns to convert"



        self.conversion_log = {}

        total_cols = len(object_cols)



        for i, col in enumerate(object_cols):

            if progress_bar:

                progress_bar.progress((i + 1) / total_cols)



            s = self.processed_df[col]

            mask = s.isna()



            if method == 'label_encoding':

                temp = s.fillna('__MISSING__').astype(str)

                le = LabelEncoder()

                le.fit(temp)

                enc = le.transform(temp).astype('int64')

                enc[mask.values] = missing_sentinel

                self.processed_df[col] = enc



                mapping = dict(zip(le.classes_, le.transform(le.classes_)))

                if '__MISSING__' in mapping:

                    mapping[''] = missing_sentinel

                    mapping.pop('__MISSING__')



                self.conversion_log[col] = mapping



            elif method == 'factorize':

                codes, uniques = pd.factorize(s)

                self.processed_df[col] = codes.astype('int64')

                self.conversion_log[col] = dict(zip(uniques, range(len(uniques))))



            elif method == 'category_codes':

                cat = s.astype('category')

                codes = cat.cat.codes

                self.processed_df[col] = codes.astype('int64')

                self.conversion_log[col] = dict(zip(cat.cat.categories, range(len(cat.cat.categories))))



        return True, f"Successfully converted {len(object_cols)} columns"

    def split_train_test(self, test_size=0.2, random_state=42, stratify_column=None):

        """Split data into train and test sets"""

        if self.processed_df is None:

            return False, "No processed data available"



        try:

            stratify_data = None

            if stratify_column and stratify_column in self.processed_df.columns:

                stratify_data = self.processed_df[stratify_column]



            self.train_df, self.test_df = train_test_split(

                self.processed_df, 

                test_size=test_size, 

                random_state=random_state,

                stratify=stratify_data

            )



            return True, f"Data split successfully: {len(self.train_df)} train, {len(self.test_df)} test samples"



        except Exception as e:

            return False, f"Error splitting data: {str(e)}"

    def get_downloadable_csv(self, dataset='processed'):

        """Generate downloadable CSV content"""

        if dataset == 'processed' and self.processed_df is not None:

            df = self.processed_df

        elif dataset == 'train' and self.train_df is not None:

            df = self.train_df

        elif dataset == 'test' and self.test_df is not None:

            df = self.test_df

        else:

            return None



        # Convert dataframe to CSV

        csv_buffer = io.StringIO()

        df.to_csv(csv_buffer, index=False)

        return csv_buffer.getvalue()

def main():

    """Main Streamlit application"""

    # Initialize session state

    if 'converter' not in st.session_state:

        st.session_state.converter = StreamlitCSVConverter()

    if 'conversion_complete' not in st.session_state:

        st.session_state.conversion_complete = False

    if 'null_handling_complete' not in st.session_state:

        st.session_state.null_handling_complete = False

    if 'split_data' not in st.session_state:

        st.session_state.split_data = False



    converter = st.session_state.converter



    # Header

    st.markdown('<h1 class="main-header">🔄 CSV Object-to-Integer Converter</h1>', unsafe_allow_html=True)

    st.markdown('<p class="sub-header">Convert categorical data for Machine Learning model training</p>', unsafe_allow_html=True)



    # Sidebar for options and information

    with st.sidebar:

        st.header("⚙️ Configuration")



        # Conversion method selection

        method = st.selectbox(

            "Encoding Method",

            ["label_encoding", "factorize", "category_codes"],

            format_func=lambda x: {

                "label_encoding": "Label Encoding (Recommended)",

                "factorize": "Pandas Factorize", 

                "category_codes": "Category Codes"

            }[x]

        )



        st.markdown("---")



        # Information section

        st.header("ℹ️ About")

        st.info("""

        This tool processes CSV files for machine learning:



        **Features:**

        - **Null Value Handling**: Multiple strategies for missing data

        - **Data Encoding**: Convert categorical to numeric data

        - **Train/Test Split**: Optional data splitting



        **Encoding Methods:**

        - **Label Encoding**: Assigns unique integers to each category

        - **Factorize**: Uses pandas factorize method

        - **Category Codes**: Converts to categorical and uses codes

        """)



        st.header("📋 Instructions")

        st.markdown("""

        1. Upload your CSV file

        2. Handle null values

        3. Choose encoding method

        4. Decide on train/test split

        5. Convert and download

        """)



    # Main content area

    col1, col2 = st.columns([1, 1])



    with col1:

        st.header("📁 File Upload")



        uploaded_file = st.file_uploader(

            "Choose a CSV file",

            type="csv",

            help="Upload a CSV file containing object/categorical columns to convert"

        )



        if uploaded_file is not None:

            # Process uploaded file

            success, error = converter.process_uploaded_file(uploaded_file)



            if success:

                st.markdown('<p class="success-message">✅ File uploaded successfully!</p>', unsafe_allow_html=True)



                # Display file information

                st.subheader("📊 File Information")

                col_info1, col_info2, col_info3 = st.columns(3)



                with col_info1:

                    st.metric("Rows", converter.original_df.shape[0])

                with col_info2:

                    st.metric("Columns", converter.original_df.shape[1])

                with col_info3:

                    object_cols = converter.original_df.select_dtypes(include=['object', 'category', 'string']).columns

                    st.metric("Object Columns", len(object_cols))



                # Analyze null values

                null_analysis = converter.analyze_null_values()



                # Show null value analysis

                st.subheader("🔍 Null Value Analysis")



                null_summary = []

                for col, analysis in null_analysis.items():

                    if analysis['null_count'] > 0:

                        null_summary.append({

                            'Column': col,

                            'Null Count': analysis['null_count'],

                            'Null %': f"{analysis['null_percentage']:.2f}%",

                            'Data Type': analysis['data_type']

                        })



                if null_summary:

                    st.dataframe(pd.DataFrame(null_summary), use_container_width=True)



                    # Null handling options

                    st.subheader("🛠️ Null Value Handling")



                    strategy_dict = {}



                    for col_info in null_summary:

                        col = col_info['Column']

                        col_type = col_info['Data Type']



                        # Determine available strategies based on data type

                        if 'int' in col_type or 'float' in col_type:

                            strategies = ['drop_rows', 'fill_mean', 'fill_median', 'fill_mode', 'fill_forward', 'fill_backward', 'fill_constant']

                        else:

                            strategies = ['drop_rows', 'fill_mode', 'fill_forward', 'fill_backward', 'fill_constant']



                        strategy = st.selectbox(

                            f"Strategy for {col}:",

                            strategies,

                            format_func=lambda x: {

                                'drop_rows': 'Drop rows with nulls',

                                'fill_mean': 'Fill with mean',

                                'fill_median': 'Fill with median', 

                                'fill_mode': 'Fill with mode (most frequent)',

                                'fill_forward': 'Forward fill',

                                'fill_backward': 'Backward fill',

                                'fill_constant': 'Fill with constant (0/Unknown)'

                            }[x],

                            key=f"strategy_{col}"

                        )



                        strategy_dict[col] = strategy



                    if st.button("🔧 Handle Null Values", type="primary", use_container_width=True):

                        with st.spinner("Handling null values..."):

                            success, message = converter.handle_null_values(strategy_dict)



                            if success:

                                st.session_state.null_handling_complete = True

                                st.success(f"✅ {message}")

                            else:

                                st.error(f"❌ {message}")



                else:

                    st.info("✅ No null values found in the dataset!")

                    st.session_state.null_handling_complete = True

                    converter.processed_df = converter.original_df.copy()



                # Data preview

                st.subheader("👀 Data Preview")

                st.dataframe(converter.original_df.head(10), use_container_width=True)



            else:

                st.markdown(f'<p class="error-message">❌ Error loading file: {error}</p>', unsafe_allow_html=True)



    with col2:

        st.header("🔄 Data Processing")



        if uploaded_file is not None and converter.original_df is not None:



            # Train/Test Split Decision

            st.subheader("🎯 Data Splitting Options")



            split_choice = st.radio(

                "How do you want to process your data?",

                ["Work as single file", "Split into train/test sets"],

                help="Choose whether to process the entire dataset or split it for ML training"

            )



            if split_choice == "Split into train/test sets":

                st.session_state.split_data = True



                col_split1, col_split2 = st.columns(2)



                with col_split1:

                    test_size = st.slider("Test size (%)", 10, 40, 20) / 100



                with col_split2:

                    random_state = st.number_input("Random state", 0, 1000, 42)



                # Stratify option

                stratify_col = st.selectbox(

                    "Stratify by column (optional)",

                    ["None"] + list(converter.original_df.columns),

                    help="Select a column to maintain its distribution in train/test split"

                )



                stratify_column = stratify_col if stratify_col != "None" else None



            else:

                st.session_state.split_data = False



            # Data conversion section

            if st.session_state.null_handling_complete:

                object_cols = converter.get_object_columns()



                if object_cols:

                    st.subheader("🔤 Object Column Conversion")

                    st.write(f"Ready to convert {len(object_cols)} object columns using **{method}** method.")



                    if st.button("🚀 Convert Data", type="primary", use_container_width=True):

                        with st.spinner("Converting data..."):

                            # Progress bar

                            progress_bar = st.progress(0)



                            # Perform conversion

                            success, message = converter.convert_objects_to_integers(method, progress_bar)



                            if success:

                                st.session_state.conversion_complete = True

                                st.success(f"✅ {message}")



                                # Handle train/test split if requested

                                if st.session_state.split_data:

                                    split_success, split_message = converter.split_train_test(

                                        test_size=test_size,

                                        random_state=random_state,

                                        stratify_column=stratify_column

                                    )



                                    if split_success:

                                        st.success(f"✅ {split_message}")

                                    else:

                                        st.error(f"❌ {split_message}")



                                # Show conversion summary

                                st.subheader("📈 Conversion Summary")

                                summary_data = []



                                for col in converter.original_df.columns:

                                    orig_type = str(converter.original_df[col].dtype)

                                    new_type = str(converter.processed_df[col].dtype)

                                    status = "Converted" if orig_type != new_type else "Unchanged"



                                    summary_data.append({

                                        "Column": col,

                                        "Original Type": orig_type,

                                        "New Type": new_type,

                                        "Status": status

                                    })



                                summary_df = pd.DataFrame(summary_data)

                                st.dataframe(summary_df, use_container_width=True)



                            else:

                                st.error(f"❌ {message}")



                else:

                    st.info("No object columns to convert. Your data is already in numeric format!")

                    st.session_state.conversion_complete = True



            else:

                st.info("👆 Please handle null values first before conversion.")



        else:

            st.info("👆 Please upload a CSV file to start processing.")



    # Results section (full width)

    if st.session_state.conversion_complete and converter.processed_df is not None:

        st.markdown("---")

        st.header("✨ Results")



        # Create tabs based on whether data is split or not

        if st.session_state.split_data and converter.train_df is not None:

            tab1, tab2, tab3, tab4 = st.tabs([

                "📊 Processed Data", 

                "🚂 Train Set", 

                "🧪 Test Set", 

                "🗺️ Conversion Mappings"

            ])

        else:

            tab1, tab2, tab3 = st.tabs([

                "📊 Processed Data", 

                "🔍 Comparison", 

                "🗺️ Conversion Mappings"

            ])



        with tab1:

            st.subheader("Processed Dataset")

            st.dataframe(converter.processed_df.head(20), use_container_width=True)



            # Data types comparison

            col_types1, col_types2 = st.columns(2)



            with col_types1:

                st.subheader("Original Data Types")

                orig_types = pd.DataFrame({

                    'Column': converter.original_df.columns,

                    'Data Type': [str(dtype) for dtype in converter.original_df.dtypes]

                })

                st.dataframe(orig_types, use_container_width=True)



            with col_types2:

                st.subheader("Processed Data Types")

                proc_types = pd.DataFrame({

                    'Column': converter.processed_df.columns,

                    'Data Type': [str(dtype) for dtype in converter.processed_df.dtypes]

                })

                st.dataframe(proc_types, use_container_width=True)



        if st.session_state.split_data and converter.train_df is not None:

            with tab2:

                st.subheader("Training Set")

                st.dataframe(converter.train_df.head(20), use_container_width=True)

                st.info(f"Training set shape: {converter.train_df.shape}")



            with tab3:

                st.subheader("Test Set")

                st.dataframe(converter.test_df.head(20), use_container_width=True)

                st.info(f"Test set shape: {converter.test_df.shape}")



            with tab4:

                st.subheader("Conversion Mappings")

                for col, mapping in converter.conversion_log.items():

                    with st.expander(f"📋 Column: {col}"):

                        mapping_df = pd.DataFrame([

                            {"Original Value": k, "Encoded Value": v}

                            for k, v in list(mapping.items())[:20]  # Show first 20 mappings

                        ])

                        st.dataframe(mapping_df, use_container_width=True)



                        if len(mapping) > 20:

                            st.info(f"Showing first 20 of {len(mapping)} total mappings.")



        else:

            with tab2:

                st.subheader("Before vs After Comparison")



                # Select column to compare

                object_cols_orig = converter.original_df.select_dtypes(include=['object', 'category', 'string']).columns



                if len(object_cols_orig) > 0:

                    selected_col = st.selectbox("Select column to compare:", object_cols_orig)



                    comp_col1, comp_col2 = st.columns(2)



                    with comp_col1:

                        st.write("**Original Values (Sample)**")

                        orig_sample = converter.original_df[selected_col].value_counts().head(10)

                        st.dataframe(orig_sample, use_container_width=True)



                    with comp_col2:

                        st.write("**Encoded Values (Sample)**")

                        proc_sample = converter.processed_df[selected_col].value_counts().head(10)

                        st.dataframe(proc_sample, use_container_width=True)



            with tab3:

                st.subheader("Conversion Mappings")



                for col, mapping in converter.conversion_log.items():

                    with st.expander(f"📋 Column: {col}"):

                        mapping_df = pd.DataFrame([

                            {"Original Value": k, "Encoded Value": v}

                            for k, v in list(mapping.items())[:20]  # Show first 20 mappings

                        ])

                        st.dataframe(mapping_df, use_container_width=True)



                        if len(mapping) > 20:

                            st.info(f"Showing first 20 of {len(mapping)} total mappings.")



        # Download section

        st.markdown("---")

        st.header("💾 Download Processed Data")



        if st.session_state.split_data and converter.train_df is not None:

            # Multiple download buttons for train/test split

            col_dl1, col_dl2, col_dl3 = st.columns(3)



            with col_dl1:

                csv_data = converter.get_downloadable_csv('processed')

                if csv_data:

                    st.download_button(

                        label="📥 Download Full Dataset",

                        data=csv_data,

                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_processed.csv",

                        mime="text/csv",

                        type="primary",

                        use_container_width=True

                    )



            with col_dl2:

                train_csv = converter.get_downloadable_csv('train')

                if train_csv:

                    st.download_button(

                        label="📥 Download Train Set",

                        data=train_csv,

                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_train.csv",

                        mime="text/csv",

                        type="secondary",

                        use_container_width=True

                    )



            with col_dl3:

                test_csv = converter.get_downloadable_csv('test')

                if test_csv:

                    st.download_button(

                        label="📥 Download Test Set",

                        data=test_csv,

                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_test.csv",

                        mime="text/csv",

                        type="secondary",

                        use_container_width=True

                    )



        else:

            # Single download button

            csv_data = converter.get_downloadable_csv('processed')

            if csv_data:

                st.download_button(

                    label="📥 Download Processed CSV",

                    data=csv_data,

                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_processed.csv",

                    mime="text/csv",

                    type="primary",

                    use_container_width=True

                )



        # File size info

        if csv_data:

            file_size = len(csv_data.encode('utf-8')) / 1024  # KB

            st.info(f"File size: {file_size:.2f} KB")

if __name__ == "__main__":

    main()
