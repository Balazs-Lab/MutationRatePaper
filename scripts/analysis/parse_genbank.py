"""
Parse GenBank files to extract CDS and misc_feature annotations.
Handles join() features for spliced CDS.
"""

from Bio import SeqIO
from typing import List, Dict, Tuple, Optional
import re


class Feature:
    """Represents a genomic feature (CDS or misc_feature)."""
    
    def __init__(self, feature_type: str, name: str, locations: List[Tuple[int, int]], 
                 gene: Optional[str] = None):
        self.feature_type = feature_type  # 'CDS' or 'misc_feature'
        self.name = name
        self.locations = locations  # List of (start, end) tuples (1-based, inclusive)
        self.gene = gene
    
    def contains_position(self, pos: int) -> bool:
        """Check if a genomic position (1-based) is within this feature."""
        for start, end in self.locations:
            if start <= pos <= end:
                return True
        return False
    
    def get_cds_position(self, genomic_pos: int) -> Optional[int]:
        """
        Convert genomic position to CDS position (1-based nucleotide position in CDS).
        Returns None if position is not in this CDS.
        For join() features, returns position in the concatenated CDS.
        """
        if not self.contains_position(genomic_pos):
            return None
        
        cds_pos = 0
        for start, end in self.locations:
            if start <= genomic_pos <= end:
                cds_pos += (genomic_pos - start) + 1
                return cds_pos
            else:
                cds_pos += (end - start) + 1
        return None
    
    def get_all_genomic_positions(self) -> List[int]:
        """Get all genomic positions covered by this feature."""
        positions = []
        for start, end in self.locations:
            positions.extend(range(start, end + 1))
        return positions


def parse_location_string(location_str: str) -> List[Tuple[int, int]]:
    """
    Parse GenBank location string into list of (start, end) tuples.
    Handles:
    - Simple: 33..590
    - Join: join(801..1015,3340..3385)
    - Complement: complement(33..590)
    """
    locations = []
    
    # Clean the string - remove any non-ASCII characters that might cause issues
    location_str = ''.join(c for c in location_str if c.isprintable() and (c.isalnum() or c in '().,..'))
    
    # Remove 'complement(' wrapper if present
    is_complement = False
    if location_str.startswith('complement('):
        is_complement = True
        location_str = location_str[11:-1]  # Remove 'complement(' and ')'
    
    # Handle join()
    if location_str.startswith('join('):
        location_str = location_str[5:-1]  # Remove 'join(' and ')'
        parts = location_str.split(',')
    else:
        parts = [location_str]
    
    for part in parts:
        part = part.strip()
        # Handle ranges like "33..590" or single positions
        if '..' in part:
            start_str, end_str = part.split('..', 1)
            # Clean any remaining non-numeric characters
            start_str = ''.join(c for c in start_str if c.isdigit())
            end_str = ''.join(c for c in end_str if c.isdigit())
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else 0
        else:
            # Single position - clean non-numeric characters
            part = ''.join(c for c in part if c.isdigit())
            start = end = int(part) if part else 0
        
        if start > 0 and end > 0:
            locations.append((start, end))
    
    return locations


def parse_genbank_file(gb_file: str) -> Tuple[str, List[Feature]]:
    """
    Parse a GenBank file and extract sequence and features.
    
    Returns:
        (sequence, features_list)
    """
    try:
        # Try using BioPython
        record = SeqIO.read(gb_file, "genbank")
        sequence = str(record.seq).upper()
        features = []
        
        for feature in record.features:
            if feature.type in ['CDS', 'misc_feature']:
                # Get name from standard_name qualifier
                name = feature.qualifiers.get('standard_name', [''])[0]
                if not name:
                    # Try gene qualifier for CDS
                    name = feature.qualifiers.get('gene', [''])[0]
                if not name:
                    name = feature.type
                
                # Get gene name if available
                gene = feature.qualifiers.get('gene', [None])[0]
                
                # BioPython uses 0-based [start:end); convert to 1-based (start, end) inclusive
                loc = feature.location
                locations = []
                if hasattr(loc, 'parts'):
                    for part in loc.parts:
                        locations.append((part.start + 1, part.end))
                else:
                    locations.append((loc.start + 1, loc.end))
                
                feat = Feature(feature.type, name, locations, gene)
                features.append(feat)
        
        return sequence, features
    
    except Exception as e:
        # Fallback: manual parsing if BioPython fails
        print(f"Warning: BioPython parsing failed ({e}), using manual parser")
        return parse_genbank_manual(gb_file)


def parse_genbank_manual(gb_file: str) -> Tuple[str, List[Feature]]:
    """
    Manual parser for GenBank files (fallback if BioPython unavailable).
    """
    with open(gb_file, 'r') as f:
        lines = f.readlines()
    
    sequence = ""
    features = []
    in_origin = False
    current_feature = None
    current_locations = None
    current_name = None
    current_gene = None
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Check for end of file first
        if line.startswith('//'):
            if in_origin:
                break
            i += 1
            continue
        
        # Extract sequence from ORIGIN section (must check before features)
        if 'ORIGIN' in line:
            in_origin = True
            i += 1
            continue
        
        if in_origin:
            # Extract sequence (remove numbers and spaces)
            # Format: "       61 ggattaacac atggaaaaga..."
            # Remove leading whitespace and line numbers
            line_stripped = line.strip()
            if line_stripped:
                # Split by whitespace, skip first element (line number), join rest
                parts = line_stripped.split()
                if len(parts) > 1:
                    # Skip the line number, join the rest
                    seq_part = ''.join(parts[1:])
                    sequence += seq_part
            i += 1
            continue
        
        # Parse qualifiers (only if not in ORIGIN section)
        if not in_origin and line.strip().startswith('/'):
            qual_line = line.strip()
            # Continue reading multi-line qualifiers
            while i + 1 < len(lines) and not lines[i + 1].strip().startswith('/') and not lines[i + 1].startswith('     ') and 'ORIGIN' not in lines[i + 1]:
                i += 1
                qual_line += lines[i].strip()
            
            if qual_line.startswith('/standard_name='):
                match = re.search(r'standard_name="([^"]+)"', qual_line)
                if match:
                    current_name = match.group(1)
            elif qual_line.startswith('/gene='):
                match = re.search(r'gene="([^"]+)"', qual_line)
                if match:
                    current_gene = match.group(1)
            
            i += 1
            continue
        
        # Skip feature parsing if in ORIGIN section
        if not in_origin:
            # Parse features
            if line.startswith('     CDS') or line.startswith('     misc_feature'):
                # Save previous feature if exists
                if current_feature and current_locations:
                    feat = Feature(current_feature, current_name or current_feature, 
                                 current_locations, current_gene)
                    features.append(feat)
                
                # Start new feature
                parts = line.split()
                current_feature = parts[0].strip()
                # Handle location string - may have special characters, extract just the location part
                # Join all parts after the feature type to handle cases where location has spaces
                location_str = ' '.join(parts[1:]) if len(parts) > 1 else ""
                current_locations = parse_location_string(location_str)
                current_name = None
                current_gene = None
                i += 1
                continue
        
        i += 1
    
    # Save last feature
    if current_feature and current_locations:
        feat = Feature(current_feature, current_name or current_feature, 
                     current_locations, current_gene)
        features.append(feat)
    
    return sequence.upper(), features


if __name__ == "__main__":
    # Test parsing
    import sys
    gb_file = sys.argv[1] if len(sys.argv) > 1 else "../../data/00-bed_files/NL43.gb"
    seq, features = parse_genbank_file(gb_file)
    print(f"Sequence length: {len(seq)}")
    print(f"Number of features: {len(features)}")
    for feat in features:
        print(f"{feat.feature_type}: {feat.name} - {feat.locations}")
