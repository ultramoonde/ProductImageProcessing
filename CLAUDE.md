# Claude Code Configuration

# CRITICAL DEVELOPMENT RULES - MANDATORY FOLLOW VERBATIM

## NEVER CREATE NEW FILES RULE
❌ **NEVER create new test files, new pipeline files, or new main files**
❌ **NEVER create files with names like: test_*, *_test.py, new_*, fixed_*, temp_***
✅ **ONLY edit existing files in the codebase**
✅ **If unsure which file to edit, ASK THE USER first**

## EXISTING PIPELINE IDENTIFICATION
**Working Pipeline Files:**
- Main pipeline: `[USER TO SPECIFY]`
- Step modules: `src/steps/step_1_ui_analysis.py` through `src/steps/step_6_csv_generation.py`
- **ONLY edit these existing files - NEVER create new ones**

## BEFORE ANY CODE CHANGES
1. **READ the existing working pipeline file first**
2. **UNDERSTAND the current data flow between steps**
3. **IDENTIFY the specific broken function/section**
4. **EDIT only that broken section**
5. **NEVER rewrite entire files or create new test frameworks**

## FORBIDDEN ACTIONS
❌ Creating any new .py files
❌ Saying "let me create a test to..."
❌ Making new pipeline versions
❌ Rewriting working code sections
❌ Testing with mock data instead of real pipeline data

## REQUIRED ACTIONS
✅ Edit existing files only
✅ Preserve all working functionality
✅ Fix only the specific reported issue
✅ Test using the existing working pipeline
✅ Ask user which file to edit if unsure

## DEBUG PROTOCOL
1. Run the existing working pipeline
2. Identify where exactly it fails
3. Edit only that specific function
4. Test with the same existing pipeline
5. **NEVER create new test files**

**VIOLATION OF THESE RULES = IMMEDIATE STOP AND ASK USER**

---

## Approved Test Images Path
**Location:** `/Users/davemooney/_dev/Flink/Approved`

This folder contains pre-approved Flink grocery app screenshots for testing the modular pipeline. These images should have proper UI structure and not cause issues with the pipeline steps.

## Pipeline Status
- ✅ Liquid unit extraction fixes implemented
- ✅ Pink button removal (60px radius) working
- ✅ CSV field mapping corrected (weight → weight_quantity)
- ⚠️ UI Analysis step has compatibility issues with some images

## Test Protocol
1. Check `/Users/davemooney/_dev/Flink/Approved` for available test images first
2. Use approved images for pipeline testing to avoid UI structure issues
3. Full 6-step modular pipeline should work with approved images

## Last Updated
2025-09-26 - Added critical development rules and approved test images path