include config.mk

# Generate plots with parameters set in config.mk.

PNG_FILES=$(wildcard *.png)

all : $(PNG_FILES) Assignment_4.pdf

Assignment_4.pdf : Assignment_4.tex $(PNG_FILES)
	$(PDF_MAKER) $<

$(PNG_FILES) : $(ODE_SOLVER)  plot_instruction_set.txt \
	err_instruction_set.txt
		while read -r file; do \
			$(ODE_EXE) "$$file" $(SETTINGS); \
			done <plot_instruction_set.txt
		while read -r errfile; do \
			$(ODE_EXE) "$$errfile" $(ERR_BEHAVIOR_SETTINGS); \
			done <err_instruction_set.txt

