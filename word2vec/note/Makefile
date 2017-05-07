OBJ := main

build: 
	latex $(OBJ)
	latex $(OBJ)

.PHONY: clean

#note that a space is needed between {} and \;
clean:
	find . -name 'main.*' ! -name '*.tex' -exec rm -rf {} \;

pdf: 
	pdflatex $(OBJ)

show: 
	xdvi $(OBJ)

png2eps:
	sh ./png2eps.sh
