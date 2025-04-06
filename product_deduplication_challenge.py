import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import os
import argparse
from collections import defaultdict

def clean_text(text):
    """Clean text for better matching"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Transform to lowercase and remove spaces
    text = text.lower().strip()
    
    # Remove unwanted characters
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

def extract_key_features(row):
    """Extract standardized key features from a product record"""
    features = {}
    
    # Extract product identifiers if available
    if 'product_id' in row and not pd.isna(row['product_id']):
        features['product_id'] = str(row['product_id']).lower().strip()
    
    if 'sku' in row and not pd.isna(row['sku']):
        features['sku'] = str(row['sku']).lower().strip()
        
    if 'upc' in row and not pd.isna(row['upc']):
        features['upc'] = str(row['upc']).lower().strip()
        
    if 'brand' in row and not pd.isna(row['brand']):
        features['brand'] = clean_text(str(row['brand']))
    
    # Create a normalized product name
    if 'product_name' in row and not pd.isna(row['product_name']):
        features['norm_name'] = clean_text(str(row['product_name']))
    elif 'product_title' in row and not pd.isna(row['product_title']):
        features['norm_name'] = clean_text(str(row['product_title']))
    
    return features

def create_blocking_key(features):
    """Create a blocking key to reduce comparison space"""
    if 'norm_name' in features and features['norm_name']:
        # Use first 3 chars of name + first char of brand if available
        name_prefix = features['norm_name'][:3] if len(features['norm_name']) >= 3 else features['norm_name']
        brand_prefix = features.get('brand', '')[:1] if features.get('brand') else ''
        return f"{name_prefix}_{brand_prefix}"
    return None

def calculate_field_quality(value, field_name):
    """Calculate a quality score for a field value"""
    # First check for NaN values
    if isinstance(value, (np.ndarray, pd.Series)):
        if value.size > 0 and pd.isna(value).any():
            if pd.isna(value):
                return 0
    
    # Handle empty strings
    if isinstance(value, str) and value.strip() == "":
        return 0
    
    # For non-string values, just return 1
    if not isinstance(value, str):
        return 1
    
    value = str(value).strip()
    if not value:
        return 0
    
    # Higher quality for longer text in descriptive fields
    if field_name in ['description', 'product_summary', 'product_name', 'product_title']:
        # Cap the length bonus to avoid extremely long but low-quality descriptions
        length_score = min(len(value) / 100, 1)
        
        # Higher quality for well-structured text with proper capitalization
        has_sentence_case = any(c.isupper() for c in value[:20])
        has_punctuation = any(c in value for c in '.,:;')
        formatting_score = 0.5 if has_sentence_case else 0
        formatting_score += 0.5 if has_punctuation else 0
        
        return (length_score * 0.7) + (formatting_score * 0.3)
    
    # For other fields, simple presence is good enough
    return 1

def preprocess_data(df):
    """Get data ready for deduplication with enhanced preprocessing"""
    print("Cleaning up and preparing the data...")
    
    clean_df = df.copy()
    
    # Identify text and key columns
    text_columns = ['product_name', 'product_title', 'description', 'product_summary']
    text_columns = [col for col in text_columns if col in df.columns]
    
    if not text_columns:
        print("Warning: Can't find standard text columns!")
        # Use first few string columns as fallback
        text_columns = [col for col in df.columns[:5] if df[col].dtype == 'object'][:3]
    
    # Clean up text data
    for col in text_columns:
        clean_df[col] = clean_df[col].fillna("")
        clean_df[col] = clean_df[col].apply(clean_text)
    
    # Combine text for similarity matching
    clean_df['all_text'] = ""
    for col in text_columns:
        # Weight more important fields higher
        if col in ['product_name', 'product_title']:
            clean_df['all_text'] += clean_df[col] + " " + clean_df[col] + " "
        else:
            clean_df['all_text'] += clean_df[col] + " "
    
    clean_df['all_text'] = clean_df['all_text'].str.strip()
    
    # Calculate quality scores for each field
    print("Calculating field quality scores...")
    for col in df.columns:
        quality_col = f"{col}_quality"
        # Use Series.apply with explicit handling of each value
        clean_df[quality_col] = df[col].apply(lambda x: calculate_field_quality(x, col))
    
    # Calculate overall record quality score
    quality_cols = [f"{col}_quality" for col in df.columns]
    clean_df['record_quality'] = clean_df[quality_cols].sum(axis=1)
    
    # Extract key features for exact matching
    print("Extracting key features...")
    features_list = []
    for idx, row in clean_df.iterrows():
        features = extract_key_features(row)
        features_list.append(features)
    
    # Add features as new columns
    for feature in ['product_id', 'sku', 'upc', 'brand', 'norm_name']:
        clean_df[f'clean_{feature}'] = [features.get(feature, "") for features in features_list]
    
    # Create blocking keys to reduce comparison space
    clean_df['blocking_key'] = [create_blocking_key(features) for features in features_list]
    
    # Keep track of original row numbers
    clean_df['orig_index'] = df.index
    
    return clean_df

def exact_match_products(df):
    """Find products that match exactly on key identifiers"""
    print("Finding exact matches on key fields...")
    
    # Dictionary to store groups of exact matches
    exact_matches = defaultdict(list)
    matched_indices = set()
    
    # Check for exact product identifiers
    id_fields = ['clean_product_id', 'clean_sku', 'clean_upc']
    id_fields = [f for f in id_fields if f in df.columns]
    
    for field in id_fields:
        # Skip empty values
        valid_mask = (df[field].notna()) & (df[field] != "")
        valid_df = df[valid_mask]
        
        # Group by exact field value
        for value, group in valid_df.groupby(field):
            if len(group) > 1:
                # Found multiple products with same identifier
                indices = group['orig_index'].tolist()
                group_key = f"{field}:{value}"
                exact_matches[group_key].extend(indices)
                matched_indices.update(indices)
    
    # Check for name + brand matches for remaining products
    if 'clean_norm_name' in df.columns and 'clean_brand' in df.columns:
        # Only consider products not already matched by ID
        unmatched_df = df[~df['orig_index'].isin(matched_indices)]
        
        # Group by name + brand
        name_brand_df = unmatched_df[(unmatched_df['clean_norm_name'] != "") & 
                                    (unmatched_df['clean_brand'] != "")]
        
        for (name, brand), group in name_brand_df.groupby(['clean_norm_name', 'clean_brand']):
            if len(group) > 1:
                # Found multiple products with same name + brand
                indices = group['orig_index'].tolist()
                group_key = f"name_brand:{name}_{brand}"
                exact_matches[group_key].extend(indices)
                matched_indices.update(indices)
    
    # Convert to list of groups
    exact_groups = [indices for indices in exact_matches.values() if len(indices) > 1]
    
    print(f"Found {len(exact_groups)} groups with exact matches ({len(matched_indices)} products)")
    return exact_groups, matched_indices

def find_similar_products(df, text_col='all_text', threshold=0.8, batch_size=1000, exclude_indices=None):
    """Find products that are similar to each other using text similarity"""
    print(f"Looking for similar products (threshold={threshold})...")
    
    similar_pairs = []
    remaining_df = df if exclude_indices is None else df[~df['orig_index'].isin(exclude_indices)]
    
    # Skip if no products to process
    if len(remaining_df) <= 1:
        return similar_pairs
    
    # Group by blocking key to reduce comparison space
    if 'blocking_key' in remaining_df.columns and remaining_df['blocking_key'].notna().any():
        print("Using blocking keys to reduce comparison space...")
        blocks = remaining_df.groupby('blocking_key')
        total_blocks = len(blocks)
        processed = 0
        
        for block_key, block_df in blocks:
            processed += 1
            if len(block_df) <= 1:
                continue
                
            if processed % 10 == 0:
                print(f"Processing block {processed}/{total_blocks}...")
            
            # Find similar pairs within this block
            block_pairs = process_similarity_batch(block_df, text_col, threshold)
            similar_pairs.extend(block_pairs)
    else:
        # Fall back to batch-based processing for all products
        total_batches = (len(remaining_df) + batch_size - 1) // batch_size
        
        for i in range(0, len(remaining_df), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{total_batches}...")
            
            # Get the current batch
            end_i = min(i + batch_size, len(remaining_df))
            batch_df = remaining_df.iloc[i:end_i]
            
            # Process this batch against all batches (including itself)
            batch_pairs = []
            for j in range(i, len(remaining_df), batch_size):
                end_j = min(j + batch_size, len(remaining_df))
                compare_df = remaining_df.iloc[j:end_j]
                
                pairs = compare_similarity_batches(batch_df, compare_df, text_col, threshold)
                batch_pairs.extend(pairs)
            
            similar_pairs.extend(batch_pairs)
    
    print(f"Found {len(similar_pairs)} similar pairs through text comparison!")
    return similar_pairs

def process_similarity_batch(df, text_col, threshold):
    """Process a single batch for similarity"""
    pairs = []
    texts = df[text_col].tolist()
    indices = df['orig_index'].tolist()
    
    # Skip empty batches
    if len(texts) <= 1 or not any(texts):
        return pairs
    
    # Create text vectors
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    try:
        vectors = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(vectors)
        
        # Find pairs that are similar
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                if sim_matrix[i, j] >= threshold:
                    pairs.append((indices[i], indices[j], sim_matrix[i, j]))
    except Exception as e:
        print(f"Error processing similarity: {str(e)}")
    
    return pairs

def compare_similarity_batches(batch1, batch2, text_col, threshold):
    """Compare two batches for similarity"""
    pairs = []
    
    texts1 = batch1[text_col].tolist()
    texts2 = batch2[text_col].tolist()
    indices1 = batch1['orig_index'].tolist()
    indices2 = batch2['orig_index'].tolist()
    
    # Skip empty batches
    if not any(texts1) or not any(texts2):
        return pairs
    
    # For same batch comparison
    if batch1 is batch2:
        return process_similarity_batch(batch1, text_col, threshold)
    
    # Create text vectors
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    try:
        # Fit on combined texts to get consistent feature space
        all_texts = texts1 + texts2
        vectorizer.fit(all_texts)
        
        # Transform each batch separately
        vectors1 = vectorizer.transform(texts1)
        vectors2 = vectorizer.transform(texts2)
        
        # Calculate cross-batch similarity
        sim_matrix = cosine_similarity(vectors1, vectors2)
        
        # Find pairs that are similar
        for i in range(len(indices1)):
            for j in range(len(indices2)):
                if sim_matrix[i, j] >= threshold:
                    pairs.append((indices1[i], indices2[j], sim_matrix[i, j]))
    except Exception as e:
        print(f"Error comparing batches: {str(e)}")
    
    return pairs

def group_similar_products(exact_groups, text_pairs):
    """Group products into clusters of duplicates combining exact and similarity matches"""
    print("Grouping similar products...")
    
    # Build a graph where similar products are connected
    graph = {}
    
    # Add exact match groups to the graph
    for group in exact_groups:
        for i in range(len(group)):
            idx1 = group[i]
            if idx1 not in graph:
                graph[idx1] = []
            
            # Connect to all other products in the group
            for j in range(len(group)):
                if i != j:
                    idx2 = group[j]
                    graph[idx1].append(idx2)
    
    # Add text similarity pairs to the graph
    for idx1, idx2, _ in text_pairs:
        if idx1 not in graph:
            graph[idx1] = []
        if idx2 not in graph:
            graph[idx2] = []
            
        graph[idx1].append(idx2)
        graph[idx2].append(idx1)
    
    # Find all connected groups using DFS
    visited = set()
    clusters = []
    
    def dfs(node, group):
        visited.add(node)
        group.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, group)
    
    # Run DFS starting from each unvisited node
    for node in graph:
        if node not in visited:
            current_group = []
            dfs(node, current_group)
            if len(current_group) > 1:  # Only care about groups with duplicates
                clusters.append(current_group)
    
    print(f"Found {len(clusters)} groups of duplicate products")
    return clusters

def select_best_record(df, cluster):
    """Select the best record from a cluster based on quality and completeness"""
    cluster_df = df.loc[cluster]
    
    # Create a scoring system for record selection
    scores = pd.Series(0.0, index=cluster)
    
    # Factor 1: Overall record quality (40%)
    if 'record_quality' in cluster_df.columns:
        quality_scores = cluster_df['record_quality']
        max_quality = quality_scores.max() if not quality_scores.empty else 1
        normalized_quality = quality_scores / max_quality if max_quality > 0 else quality_scores
        scores += normalized_quality * 0.4
    
    # Factor 2: Non-null field count (30%)
    non_null_counts = cluster_df.notna().sum(axis=1)
    max_non_nulls = non_null_counts.max() if not non_null_counts.empty else 1
    normalized_non_nulls = non_null_counts / max_non_nulls if max_non_nulls > 0 else non_null_counts
    scores += normalized_non_nulls * 0.3
    
    # Factor 3: Text field quality (20%)
    text_columns = ['product_name', 'product_title', 'description', 'product_summary']
    text_columns = [col for col in text_columns if col in df.columns]
    
    if text_columns:
        text_quality = pd.Series(0.0, index=cluster)
        for col in text_columns:
            # Check if quality score is available
            quality_col = f"{col}_quality"
            if quality_col in cluster_df.columns:
                text_quality += cluster_df[quality_col]
            else:
                # Fallback to length-based score
                text_quality += cluster_df[col].apply(lambda x: len(str(x)) if pd.notna(x) else 0) / 100
        
        max_text_quality = text_quality.max() if not text_quality.empty else 1
        normalized_text_quality = text_quality / max_text_quality if max_text_quality > 0 else text_quality
        scores += normalized_text_quality * 0.2
    
    # Factor 4: Recency if timestamp is available (10%)
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_columns:
        try:
            # Convert first available date column to datetime
            date_col = date_columns[0]
            dates = pd.to_datetime(cluster_df[date_col], errors='coerce')
            if not dates.isna().all():
                # More recent is better
                min_date = dates.min()
                date_range = (dates.max() - min_date).total_seconds()
                if date_range > 0:
                    date_scores = ((dates - min_date).dt.total_seconds() / date_range)
                    scores += date_scores * 0.1
        except:
            # If date conversion fails, ignore this factor
            pass
    
    # Return the index with the highest score
    return scores.idxmax() if not scores.empty else cluster[0]

def merge_duplicate_fields(row_to_keep, duplicate_rows, df):
    """Merge information from duplicate rows into the main record"""
    merged_row = row_to_keep.copy()
    
    # For each column
    for col in df.columns:
        # Skip metadata columns
        if col in ['orig_index', 'all_text', 'record_quality', 'blocking_key'] or col.endswith('_quality'):
            continue
            
        # If main record has null/empty value, try to fill from duplicates
        if isinstance(merged_row[col], (np.ndarray, pd.Series)):
            if merged_row[col].size > 0 and pd.isna(merged_row[col]).any():
                if pd.isna(merged_row[col]) or (isinstance(merged_row[col], str) and merged_row[col].strip() == ''):
                    # Look for non-null values in duplicates
                    for dup_idx in duplicate_rows:
                        dup_value = df.loc[dup_idx, col]
                        if not pd.isna(dup_value) and not (isinstance(dup_value, str) and dup_value.strip() == ''):
                            merged_row[col] = dup_value
                            break
    
    return merged_row

def consolidate_duplicates(df, clusters):
    """Keep the best product from each cluster and enrich it with information from duplicates"""
    print("Consolidating duplicate products...")
    
    # Create a new DataFrame for the consolidated records
    consolidated_records = []
    
    # Process each cluster
    for cluster in clusters:
        # Select the best record
        best_idx = select_best_record(df, cluster)
        best_row = df.loc[best_idx]
        
        # Get other records in the cluster
        other_indices = [idx for idx in cluster if idx != best_idx]
        
        # Merge fields from duplicates into the best record
        if other_indices:
            merged_row = merge_duplicate_fields(best_row, other_indices, df)
            consolidated_records.append(merged_row)
        else:
            consolidated_records.append(best_row)
    
    # Add all non-duplicate records
    all_duplicate_indices = set(idx for cluster in clusters for idx in cluster)
    non_duplicate_indices = set(df.index) - all_duplicate_indices
    
    for idx in non_duplicate_indices:
        consolidated_records.append(df.loc[idx])
    
    # Create the final DataFrame
    result_df = pd.DataFrame(consolidated_records)
    
    # Clean up temporary columns
    cols_to_drop = ['all_text', 'record_quality', 'blocking_key', 'orig_index']
    cols_to_drop += [col for col in result_df.columns if col.endswith('_quality')]
    cols_to_drop += [col for col in result_df.columns if col.startswith('clean_')]
    cols_to_drop = [col for col in cols_to_drop if col in result_df.columns]
    
    if cols_to_drop:
        result_df = result_df.drop(columns=cols_to_drop)
    
    return result_df

def deduplicate_products(df, threshold=0.8, batch_size=1000):
    """Main function to deduplicate products with enhanced approach"""
    start_time = time.time()
    print("\n=== PRODUCT DEDUPLICATION ===")
    print(f"Starting with {len(df)} products")
    
    # Step 1: Clean up the data with enhanced preprocessing
    processed_df = preprocess_data(df)
    
    # Step 2: Find exact matches first
    exact_groups, exact_matched_indices = exact_match_products(processed_df)
    
    # Step 3: Find similar products using text similarity
    similar_pairs = find_similar_products(
        processed_df, 
        'all_text', 
        threshold, 
        batch_size,
        exclude_indices=exact_matched_indices
    )
    
    # Step 4: Group into clusters considering both exact and similarity matches
    clusters = group_similar_products(exact_groups, similar_pairs)
    
    if not clusters:
        print("No duplicates found!")
        return df
    
    # Step 5: Consolidate duplicates with field-level merging
    result_df = consolidate_duplicates(df, clusters)
    
    # Print results
    end_time = time.time()
    print("\n=== RESULTS ===")
    print(f"Original record count: {len(df)}")
    print(f"After deduplication: {len(result_df)}")
    print(f"Duplicates removed: {len(df) - len(result_df)}")
    print(f"Reduction: {(len(df) - len(result_df)) / len(df) * 100:.1f}%")
    print(f"Finished in {end_time - start_time:.1f} seconds")
    
    return result_df

def main():
    """Run the enhanced deduplication tool with command-line arguments"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Enhanced product deduplication tool')
    
    # Required argument
    parser.add_argument('input_file', type=str, help='Input parquet file path')
    
    # Optional arguments
    parser.add_argument('-o', '--output', type=str, help='Output file path (default: input_file_deduped.parquet)')
    parser.add_argument('-t', '--threshold', type=float, default=0.8, help='Text similarity threshold (0.0-1.0, default: 0.8)')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help='Processing batch size (default: 1000)')
    parser.add_argument('--stats', action='store_true', help='Generate deduplication statistics')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Set default output file if not specified
    output_file = args.output
    if not output_file:
        output_file = args.input_file.replace('.parquet', '') + '_deduped.parquet'
    
    # Load the data
    print(f"Loading data from {args.input_file}...")
    try:
        df = pd.read_parquet(args.input_file)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return
    
    # Run deduplication
    result_df = deduplicate_products(df, args.threshold, args.batch_size)
    
    # Save the results
    print(f"Saving deduplicated data to {output_file}...")
    result_df.to_parquet(output_file)
    
    # Generate statistics if requested
    if args.stats:
        stats_file = output_file.replace('.parquet', '_stats.csv')
        print(f"Generating deduplication statistics to {stats_file}...")
        
        stats = [
            {'metric': 'Original Records', 'value': len(df)},
            {'metric': 'Deduplicated Records', 'value': len(result_df)},
            {'metric': 'Duplicates Removed', 'value': len(df) - len(result_df)},
            {'metric': 'Reduction Percentage', 'value': f"{(len(df) - len(result_df)) / len(df) * 100:.1f}%"}
        ]
        
        pd.DataFrame(stats).to_csv(stats_file, index=False)

if __name__ == "__main__":
    main()