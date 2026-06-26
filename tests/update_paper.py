import os
import json

def update_paper(results_path="tests/results/evaluation_results.json", 
                 tex_path="paper/paper.tex", 
                 md_path="paper/paper.md"):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    # 1. Format Markdown table
    md_table = "| Method | Mean Survival (Days) | Orbits Survived | Survived 5d (%) | Final SoC (%) | Final Fuel (%) | Target Images | SAA Violations |\n"
    md_table += "|---|---|---|---|---|---|---|---|\n"
    
    for r in results:
        policy_label = {
            "passive": "No-op (Passive)",
            "random": "Random Policy",
            "rule_based": "Rule-Based Heuristic",
            "ippo": "IPPO (Baseline)",
            "mappo": "**S-MAS (MAPPO, Ours)**"
        }.get(r["policy"], r["policy"].upper())
        
        md_table += (f"| {policy_label} | {r['mean_days']:.2f} $\\pm$ {r['std_days']:.2f} | "
                     f"{r['mean_orbits']:.1f} | {r['survival_5d_pct']:.1f}\\% | "
                     f"{r['mean_soc']:.1f}\\% | {r['mean_fuel']:.1f}\\% | "
                     f"{r['mean_targets']:.1f} | {r['mean_violations']:.1f} |\n")

    # 2. Format LaTeX table
    tex_table = "\\begin{table*}[htbp]\n"
    tex_table += "\\caption{Comparative Evaluation Across 30 Seeds (5-Day Max Duration)}\n"
    tex_table += "\\label{tab:baselines}\n"
    tex_table += "\\centering\n"
    tex_table += "\\begin{tabular}{lccccccc}\n"
    tex_table += "\\toprule\n"
    tex_table += "Method & Mean Survival (Days) & Orbits Survived & Survived 5d (\\%) & Final SoC (\\%) & Final Fuel (\\%) & Target Images & SAA Violations \\\\\n"
    tex_table += "\\midrule\n"
    
    for r in results:
        policy_label = {
            "passive": "No-op (Passive)",
            "random": "Random Policy",
            "rule_based": "Rule-Based Heuristic",
            "ippo": "IPPO (Baseline)",
            "mappo": "\\textbf{S-MAS (MAPPO, Ours)}"
        }.get(r["policy"], r["policy"].upper())
        
        tex_table += (f"{policy_label} & {r['mean_days']:.2f} $\\pm$ {r['std_days']:.2f} & "
                      f"{r['mean_orbits']:.1f} & {r['survival_5d_pct']:.1f}\\% & "
                      f"{r['mean_soc']:.1f}\\% & {r['mean_fuel']:.1f}\\% & "
                      f"{r['mean_targets']:.1f} & {r['mean_violations']:.1f} \\\\\n")
                      
    tex_table += "\\bottomrule\n"
    tex_table += "\\end{tabular}\n"
    tex_table += "\\end{table*}"

    # 3. Update paper.md
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
            
        # Replace the placeholder table
        table_start = md_content.find("| Method | Mean Survival")
        if table_start != -1:
            table_end = md_content.find("|", md_content.find("SAA Violations |", table_start) + 1)
            # Find the end of that table block (a blank line or next section)
            block_end = md_content.find("\n\n", table_start)
            if block_end == -1:
                block_end = len(md_content)
                
            new_md_content = md_content[:table_start] + md_table + md_content[block_end:]
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            print(f"Updated Markdown paper at: {md_path}")
            
    # 4. Update paper.tex
    if os.path.exists(tex_path):
        with open(tex_path, "r", encoding="utf-8") as f:
            tex_content = f.read()
            
        placeholder = "*(Evaluation results to be updated automatically in final rendering).*"
        if placeholder in tex_content:
            new_tex_content = tex_content.replace(placeholder, tex_table)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(new_tex_content)
            print(f"Updated LaTeX paper at: {tex_path}")

if __name__ == "__main__":
    update_paper()
