if length == 0 then "" else
  ["<memory-context>", "Relevant memories from previous sessions:"] +
  [.[] | .payload as $p | select(($p.data // "") != "") |
    (($p.sourced_at // $p.updated_at // $p.created_at // "")[:16] | gsub("T";" ")) as $ts |
    ([$ts, ($p.project // ""), ($p.category // "")] | map(select(. != "")) | map("[\(.)]")) as $tags |
    "- \($tags | join(" ")) \($p.data)"] +
  ["</memory-context>"] | join("\n")
end
