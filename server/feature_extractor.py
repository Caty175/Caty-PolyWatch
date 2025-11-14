#!/usr/bin/env python3
"""
Feature Extractor for PE Files
Extracts the same metadata features used in training the Random Forest model.
This module uses pefile to parse PE files and extract polymorphism-relevant features.
"""

import pefile
import os
import json
from typing import Dict, Any, List
import numpy as np


class PEFeatureExtractor:
    """
    Extract metadata features from PE files that match the training dataset features.
    These features are relevant for polymorphic malware detection.
    """
    
    def __init__(self, feature_list_path: str = None):
        """
        Initialize the feature extractor.
        
        Args:
            feature_list_path: Path to the JSON file containing the ordered feature list
        """
        self.feature_list = None
        if feature_list_path and os.path.exists(feature_list_path):
            with open(feature_list_path, 'r') as f:
                metadata = json.load(f)
                self.feature_list = metadata.get('features', [])
                print(f"✅ Loaded {len(self.feature_list)} features from {feature_list_path}")
    
    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all metadata features from a PE file.

        Args:
            file_path: Path to the PE file

        Returns:
            Dictionary of feature names and values
        """
        try:
            pe = pefile.PE(file_path)
            features = {}

            # Extract all feature categories (only metadata features)
            features.update(self._extract_general_features(pe, file_path))
            features.update(self._extract_header_features(pe))
            features.update(self._extract_section_features(pe))

            pe.close()

            return features

        except Exception as e:
            print(f"❌ Error extracting features from {file_path}: {e}")
            return self._get_default_features()
    
    def _extract_general_features(self, pe: pefile.PE, file_path: str) -> Dict[str, Any]:
        """Extract general file metadata."""
        features = {}

        try:
            # File size
            features['general.size'] = os.path.getsize(file_path)

            # Virtual size
            features['general.vsize'] = pe.OPTIONAL_HEADER.SizeOfImage if hasattr(pe, 'OPTIONAL_HEADER') else 0

            # Debug information
            features['general.has_debug'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') else 0

            # Exports and imports count
            features['general.exports'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT') else 0
            features['general.imports'] = len(pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0

            # Relocations
            features['general.has_relocations'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0

            # Resources
            features['general.has_resources'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') else 0

            # Signature
            features['general.has_signature'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY') else 0

            # TLS
            features['general.has_tls'] = 1 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0

            # Symbols (usually 0 for stripped binaries)
            features['general.symbols'] = 0

        except Exception as e:
            print(f"⚠️ Error extracting general features: {e}")

        return features
    
    def _extract_header_features(self, pe: pefile.PE) -> Dict[str, Any]:
        """Extract PE header information."""
        features = {}

        try:
            # COFF Header
            if hasattr(pe, 'FILE_HEADER'):
                features['header.coff.timestamp'] = pe.FILE_HEADER.TimeDateStamp
                features['header.coff.machine'] = pe.FILE_HEADER.Machine
                features['header.coff.characteristics'] = pe.FILE_HEADER.Characteristics

            # Optional Header
            if hasattr(pe, 'OPTIONAL_HEADER'):
                opt = pe.OPTIONAL_HEADER
                features['header.optional.subsystem'] = opt.Subsystem
                features['header.optional.dll_characteristics'] = opt.DllCharacteristics
                features['header.optional.magic'] = opt.Magic
                features['header.optional.major_image_version'] = opt.MajorImageVersion
                features['header.optional.minor_image_version'] = opt.MinorImageVersion
                features['header.optional.major_linker_version'] = opt.MajorLinkerVersion
                features['header.optional.minor_linker_version'] = opt.MinorLinkerVersion
                features['header.optional.major_operating_system_version'] = opt.MajorOperatingSystemVersion
                features['header.optional.minor_operating_system_version'] = opt.MinorOperatingSystemVersion
                features['header.optional.major_subsystem_version'] = opt.MajorSubsystemVersion
                features['header.optional.minor_subsystem_version'] = opt.MinorSubsystemVersion
                features['header.optional.sizeof_code'] = opt.SizeOfCode
                features['header.optional.sizeof_headers'] = opt.SizeOfHeaders
                features['header.optional.sizeof_heap_commit'] = opt.SizeOfHeapCommit

        except Exception as e:
            print(f"⚠️ Error extracting header features: {e}")

        return features
    
    def _extract_section_features(self, pe: pefile.PE) -> Dict[str, Any]:
        """Extract section information (aggregated stats)."""
        features = {}

        try:
            if hasattr(pe, 'sections'):
                # Note: EMBER uses 'section.entry' and 'section.sections'
                features['section.entry'] = len(pe.sections)
                features['section.sections'] = len(pe.sections)
            else:
                features['section.entry'] = 0
                features['section.sections'] = 0

        except Exception as e:
            print(f"⚠️ Error extracting section features: {e}")
            features['section.entry'] = 0
            features['section.sections'] = 0

        return features
    

    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features (all zeros) in case of extraction failure."""
        if self.feature_list:
            return {feature: 0 for feature in self.feature_list}
        else:
            # Return a basic set of features with zeros
            return {}
    
    def extract_features_as_array(self, file_path: str) -> np.ndarray:
        """
        Extract features and return as a numpy array in the correct order.
        This is the format expected by the trained model.
        
        Args:
            file_path: Path to the PE file
            
        Returns:
            Numpy array of feature values in the correct order
        """
        features_dict = self.extract_features(file_path)
        
        if not self.feature_list:
            raise ValueError("Feature list not loaded. Cannot ensure correct feature order.")
        
        # Create array with features in the correct order
        feature_array = []
        for feature_name in self.feature_list:
            value = features_dict.get(feature_name, 0)
            # Handle any non-numeric values
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.0
            feature_array.append(value)
        
        return np.array(feature_array).reshape(1, -1)
    
    def extract_features_batch(self, file_paths: List[str]) -> np.ndarray:
        """
        Extract features from multiple files.
        
        Args:
            file_paths: List of paths to PE files
            
        Returns:
            Numpy array of shape (n_files, n_features)
        """
        feature_arrays = []
        for file_path in file_paths:
            features = self.extract_features_as_array(file_path)
            feature_arrays.append(features)
        
        return np.vstack(feature_arrays)


# Convenience function for quick feature extraction
def extract_pe_features(file_path: str, feature_list_path: str = None) -> Dict[str, Any]:
    """
    Quick function to extract features from a PE file.
    
    Args:
        file_path: Path to the PE file
        feature_list_path: Optional path to feature list JSON
        
    Returns:
        Dictionary of features
    """
    extractor = PEFeatureExtractor(feature_list_path)
    return extractor.extract_features(file_path)

