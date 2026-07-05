import os
import re
import sys

def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

def write_readme(content):
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)

def step1():
    os.makedirs('details', exist_ok=True)
    items = [
        ("The Plain Sequential Processing Era (Traditional ConvNets, Pre-2015)", "plain-sequential.md", "Plain Sequential Networks", "A --> B --> C"),
        ("The Element-Wise Residual Addition Era (Vanilla ResNet, 2015)", "residual-addition.md", "Residual Skip Connections", "A --> B\nA --> C\nB --> C"),
        ("The Cross-Layer Feature Concatenation Era (DenseNet, 2016–2020)", "dense-concatenation.md", "Dense Connectivity", "A --> B\nA --> C\nA --> D\nB --> C\nB --> D\nC --> D"),
        ("The Unified Low-Rank Parallel attention Era (~2023–Present)", "parallel-attention.md", "Parallel Feature Fusion", "A --> Attention\nB --> Attention"),
        ("A. Vanilla Dense Blocks (Channel-Wise Concatenation)", "vanilla-dense-blocks.md", "Vanilla Dense Blocks", "Layer1 --> Concat1\nConcat1 --> Layer2"),
        ("B. DenseNet-BC (Bottleneck & Compression)", "densenet-bc.md", "DenseNet-BC", "Concat --> 1x1Conv --> 3x3Conv"),
        ("C. Symmetrical Dense Encoder-Decoder Links (U-Net Dense Extensions)", "unet-extensions.md", "U-Net Dense Links", "Encoder --> Decoder"),
        ("D. Multi-Scale Dense Networks (MSDN)", "msdn.md", "Multi-Scale Dense Networks", "Scale1 --> Scale2\nScale1 --> Scale1"),
        ("Growth Rate ($k$) Schedulers", "growth-rate.md", "Growth Rate Schedulers", "Input --> +k --> +k --> +k"),
        ("Fused Concatenation Pointers", "fused-pointers.md", "Fused Concatenation Pointers", "MemoryAddress --> Pointer"),
        ("The Activation Memory Wall & High-Bandwidth VRAM Consumption", "memory-wall.md", "Memory Wall & VRAM", "VRAM --> OutOfMemory"),
        ("The Non-Contiguous Memory Fragment Core Stall", "fragment-stall.md", "Core Stall", "Memory --> Latency"),
        ("High-Resolution Clinical Diagnostic Volumetric Tracking (MedTech)", "medtech.md", "MedTech Applications", "MRI --> DenseNet --> Segmentation"),
        ("Real-Time Autonomous Microcontroller Computer Vision (TinyML Edge)", "tinyml.md", "TinyML Edge", "Camera --> Microcontroller --> Output"),
        ("Satellite Remote Sensing & Hyperspectral Geo-Zoning", "satellite.md", "Satellite Geo-Zoning", "SatelliteImage --> DenseNet --> Map")
    ]
    
    readme = read_readme()
    
    for title, filename, short, mermaid_inner in items:
        content = f"""# {title}

## Detailed Information
This page provides more in-depth information about **{short}**.

## Architecture Diagram
```mermaid
flowchart TD
    {mermaid_inner}
```

[Back to Main README](../README.md)
"""
        with open(os.path.join('details', filename), 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Replace title with link in README
        readme = readme.replace(f"**{title}**", f"**[{title}](details/{filename})**")
        # In case it's in a table without bold or with bold:
        # Tables might have `**The Plain...**` or just `The Plain...`
        # We need to make sure we replace the first column.
        # Let's use regex to find the exact table cell and wrap it.
    
    # Actually, the titles in the markdown are inside tables, often like `**The Plain Sequential Processing Era (Traditional ConvNets, Pre-2015)**`
    write_readme(readme)

def step2():
    os.makedirs('assets', exist_ok=True)
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200">
    <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#1a2a6c;stop-opacity:1" />
            <stop offset="50%" style="stop-color:#b21f1f;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#fdbb2d;stop-opacity:1" />
        </linearGradient>
    </defs>
    <rect width="800" height="200" fill="url(#grad1)" rx="15" />
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="40" font-weight="bold" fill="white">
        Awesome Dense Connections 🚀
        <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite" />
    </text>
    <circle cx="100" cy="100" r="30" fill="white" opacity="0.3">
        <animate attributeName="r" values="30;40;30" dur="1.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="700" cy="100" r="30" fill="white" opacity="0.3">
        <animate attributeName="r" values="30;40;30" dur="1.5s" repeatCount="indefinite"/>
    </circle>
</svg>'''
    with open('assets/banner.svg', 'w', encoding='utf-8') as f:
        f.write(svg)
    
    readme = read_readme()
    readme = readme.replace("# Awesome-Dense-Connections", "# 🧠 Awesome-Dense-Connections\n\n![Banner](assets/banner.svg)")
    readme = readme.replace("## Dense Connections in AI", "## 🌐 Dense Connections in AI: History, Progression, Variants, & Applications")
    readme = readme.replace("## 1. The Macro Chronological Evolution", "## 🕰️ 1. The Macro Chronological Evolution")
    readme = readme.replace("## 2. Core Functional & Structural Variants", "## 🏗️ 2. Core Functional & Structural Variants")
    readme = readme.replace("## 3. The Dense Block Execution Matrix", "## ⚙️ 3. The Dense Block Execution Matrix")
    readme = readme.replace("## 4. Production Engineering Challenges", "## 🛠️ 4. Production Engineering Challenges & Hardware Solutions")
    readme = readme.replace("## 5. Frontier Real-World AI Industrial Applications", "## 🌍 5. Frontier Real-World AI Industrial Applications")
    write_readme(readme)

def step3():
    readme = read_readme()
    # Add badges to the left of existing badges (if any) or create a badges section under the banner
    left_badges = '<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>'
    
    # insert after Banner
    if "![Banner](assets/banner.svg)" in readme:
        readme = readme.replace("![Banner](assets/banner.svg)", f"![Banner](assets/banner.svg)\n\n<div align=\"center\">\n{left_badges}\n</div>\n")
    else:
        readme = readme.replace("# 🧠 Awesome-Dense-Connections\n", f"# 🧠 Awesome-Dense-Connections\n\n<div align=\"center\">\n{left_badges}\n</div>\n")
    
    # SEO description
    seo_desc = "<!-- SEO Meta Description: A curated list of resources, history, variants, and applications of Dense Connections in Deep Learning and AI. Explore DenseNets, execution matrices, and frontier applications. -->\n"
    if seo_desc not in readme:
        readme = seo_desc + readme

    write_readme(readme)

def step4():
    readme = read_readme()
    right_badge = '<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>'
    # Find the badges div and append to it
    readme = readme.replace('alt="Discord" /></a>\n</div>', f'alt="Discord" /></a>{right_badge}\n</div>')
    write_readme(readme)

def step5():
    readme = read_readme()
    star_history = """
##  Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2FAwesome-Dense-Connections&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Dense-Connections&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Dense-Connections&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Dense-Connections&type=date&legend=bottom-right" />
</picture>
</a>
</div>
"""
    readme += star_history
    write_readme(readme)

def step6():
    readme = read_readme()
    readme = readme.replace("chartrepos", "chart?repos")
    write_readme(readme)

def step7():
    readme = read_readme()
    readme = readme.replace("https://github.com/sindresorhus/awesome", "https://github.com/ishandutta2007/Awesome-Awesome-Awesome")
    write_readme(readme)

if __name__ == "__main__":
    step = sys.argv[1]
    if step == "1": step1()
    elif step == "2": step2()
    elif step == "3": step3()
    elif step == "4": step4()
    elif step == "5": step5()
    elif step == "6": step6()
    elif step == "7": step7()
