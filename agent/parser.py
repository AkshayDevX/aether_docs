import os
import re
from typing import List, Dict, Any

class HierarchicalMarkdownParser:
    """
    Parses markdown documentation hierarchically by headers.
    Ensures structural integrity (keeping lists, code blocks, and sub-sections cohesive).
    """
    
    @staticmethod
    def parse_file(file_path: str) -> List[Dict[str, Any]]:
        """
        Reads a markdown file and splits it into structural chunks based on heading tokens.
        Each chunk is returned with content, source path, and primary section heading context.
        """
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            return []
            
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        file_name = os.path.basename(file_path)
        chunks = []
        
        current_header = "Introduction"
        current_chunk_lines = []
        
        # Regex to detect markdown headers: #, ##, ###, ####, etc.
        header_regex = re.compile(r"^(#{1,6})\s+(.*)$")
        
        for line in lines:
            header_match = header_regex.match(line)
            if header_match:
                # If we hit a new header and have accumulated content, save the previous chunk
                if current_chunk_lines:
                    content = "".join(current_chunk_lines).strip()
                    if content: # only append non-empty content
                        chunks.append({
                            "content": content,
                            "source": file_name,
                            "heading": current_header
                        })
                
                # Reset tracking for new section
                current_header = header_match.group(2).strip()
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)
                
        # Append the final remaining chunk
        if current_chunk_lines:
            content = "".join(current_chunk_lines).strip()
            if content:
                chunks.append({
                    "content": content,
                    "source": file_name,
                    "heading": current_header
                })
                
        # If no markdown headings were found at all, return the entire file as a single chunk
        if not chunks and lines:
            chunks.append({
                "content": "".join(lines).strip(),
                "source": file_name,
                "heading": "Full Document"
            })
            
        # Post-processing: If a chunk is exceptionally large (e.g. > 4000 characters),
        # sub-divide it by double line breaks to keep context size tight.
        processed_chunks = []
        for c in chunks:
            if len(c["content"]) > 4000:
                sub_chunks = HierarchicalMarkdownParser._subdivide_chunk(c)
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(c)
                
        return processed_chunks

    @staticmethod
    def _subdivide_chunk(chunk: Dict[str, Any], max_chars: int = 3000) -> List[Dict[str, Any]]:
        """Helper to break down massive sections into smaller paragraph blocks while retaining metadata."""
        content = chunk["content"]
        paragraphs = content.split("\n\n")
        
        sub_chunks = []
        current_sub_lines = []
        current_len = 0
        part_idx = 1
        
        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > max_chars and current_sub_lines:
                # Flush current sub-chunk
                sub_content = "\n\n".join(current_sub_lines)
                sub_chunks.append({
                    "content": sub_content,
                    "source": chunk["source"],
                    "heading": f"{chunk['heading']} (Part {part_idx})"
                })
                part_idx += 1
                current_sub_lines = [para]
                current_len = para_len
            else:
                current_sub_lines.append(para)
                current_len += para_len + 2 # account for double newline separator
                
        if current_sub_lines:
            sub_content = "\n\n".join(current_sub_lines)
            sub_chunks.append({
                "content": sub_content,
                "source": chunk["source"],
                "heading": f"{chunk['heading']} (Part {part_idx})"
            })
            
        return sub_chunks
