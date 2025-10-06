
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

Usage:
    streamlit run csv_converter_streamlit.py

Requirements: streamlit, pandas, scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitCSVConverter:
    """Main class for Streamlit CSV converter"""

    def __init__(self):
        self.original_df = None
        self.processed_df = None
        self.conversion_log = {}

    def process_uploaded_file(self, uploaded_file):
        """Process uploaded CSV file"""
        try:
            # Read CSV from uploaded file
            self.original_df = pd.read_csv(uploaded_file)
            return True, None
        # If file not read, throw an exception and return False
        except Exception as e:
            return False, str(e)

    def get_object_columns(self):
        """Get list of object columns"""
        if self.original_df is None:
            return []
        return self.original_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    def convert_objects_to_integers(self, method='label_encoding', progress_bar=None, missing_sentinel=-1):
        if self.original_df is None:
            return False, "No data loaded"

        object_cols = self.get_object_columns()
        if not object_cols:
            return True, "No object columns to convert"

        self.processed_df = self.original_df.copy()
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
                    mapping['<MISSING>'] = missing_sentinel
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

    def get_downloadable_csv(self):
        """Generate downloadable CSV content"""
        if self.processed_df is None:
            return None

        # Convert dataframe to CSV
        csv_buffer = io.StringIO()
        self.processed_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

def main():
    """Main Streamlit application"""
    # Initialize session state
    if 'converter' not in st.session_state:
        st.session_state.converter = StreamlitCSVConverter()

    if 'conversion_complete' not in st.session_state:
        st.session_state.conversion_complete = False

    converter = st.session_state.converter

    # Header
    st.markdown('<div class="main-header">🔄 CSV Object-to-Integer Converter</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Convert categorical data for Machine Learning model training</div>', unsafe_allow_html=True)

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
        This tool converts object (categorical) columns in your CSV file to integer format, 
        making your data ready for machine learning models.

        **Encoding Methods:**
        - **Label Encoding**: Assigns unique integers to each category
        - **Factorize**: Uses pandas factorize method
        - **Category Codes**: Converts to categorical and uses codes
        """)

        st.header("📋 Instructions")
        st.markdown("""
        1. Upload your CSV file
        2. Choose an encoding method
        3. Click 'Convert Data'
        4. Download the processed file
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
                st.markdown('<div class="success-box">✅ File uploaded successfully!</div>', unsafe_allow_html=True)

                # Display file information
                st.subheader("📊 File Information")
                col_info1, col_info2, col_info3 = st.columns(3)

                with col_info1:
                    st.metric("Rows", converter.original_df.shape[0])

                with col_info2:
                    st.metric("Columns", converter.original_df.shape[1])

                with col_info3:
                    object_cols = converter.get_object_columns()
                    st.metric("Object Columns", len(object_cols))

                # Show object columns
                if object_cols:
                    st.subheader("🏷️ Object Columns Found")
                    for col in object_cols:
                        unique_count = converter.original_df[col].nunique()
                        st.write(f"• **{col}**: {unique_count} unique values")
                else:
                    st.info("No object columns found in the dataset. Data appears to be already numeric.")

                # Data preview
                st.subheader("👀 Data Preview")
                st.dataframe(converter.original_df.head(10), use_container_width=True)

            else:
                st.markdown(f'<div class="error-box">❌ Error loading file: {error}</div>', unsafe_allow_html=True)

    with col2:
        st.header("🔄 Data Conversion")

        if uploaded_file is not None and converter.original_df is not None:
            object_cols = converter.get_object_columns()

            if object_cols:
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
        else:
            st.info("👆 Please upload a CSV file to start conversion.")

    # Results section (full width)
    if st.session_state.conversion_complete and converter.processed_df is not None:
        st.markdown("---")
        st.header("✨ Results")

        # Tabs for comparison
        tab1, tab2, tab3 = st.tabs(["📊 Processed Data", "🔍 Comparison", "🗺️ Conversion Mappings"])

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

        with tab2:
            st.subheader("Before vs After Comparison")

            # Select column to compare
            object_cols = converter.get_object_columns()
            if object_cols:
                selected_col = st.selectbox("Select column to compare:", object_cols)

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

        csv_data = converter.get_downloadable_csv()
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
            file_size = len(csv_data.encode('utf-8')) / 1024  # KB
            st.info(f"File size: {file_size:.2f} KB")

if __name__ == "__main__":
    main()
