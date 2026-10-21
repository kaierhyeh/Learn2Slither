SHELL   := /bin/bash
NAME     = snake
SRCDIR   = src
TESTDIR  = tests
MAIN     = $(SRCDIR)/main.py
PYTHON   = $(shell [ -x venv/bin/python3 ] && echo venv/bin/python3 || echo python3)
PIP      = $(shell [ -x venv/bin/pip3 ] && echo venv/bin/pip3 || echo pip3)

# ========================================================
# Colors and Formatting
# ========================================================
RESET_LINE  = \033[2K\r
RED         = \033[1;31m
RED_FLASH   = \033[5;31m
YELLOW      = \033[1;33m
GREEN       = \033[1;32m
CYAN        = \033[1;36m
BLUE        = \033[1;34m
BLUE_LIGHT  = \033[1;36m
GREY        = \033[1;30m
NOCOLOR     = \033[0m

# ========================================================
# Draw progress bar
# Usage: draw_bar <message> <color> <icon> [delay_seconds]
# ========================================================
DRAW_BAR = draw_bar() { \
	local msg=$$1 color=$$2 icon=$$3 delay=$${4:-0.06}; \
	local bars=( \
		"[          ]   0%" \
		"[█         ]  10%" \
		"[██        ]  20%" \
		"[███       ]  30%" \
		"[████      ]  40%" \
		"[█████     ]  50%" \
		"[██████    ]  60%" \
		"[███████   ]  70%" \
		"[████████  ]  80%" \
		"[█████████ ]  90%" \
		"[██████████] 100%" \
	); \
	for b in "$${bars[@]}"; do \
		printf "$(RESET_LINE)$${color}→ $${icon}  %-24s %s$(NOCOLOR)" "$$msg" "$$b"; \
		sleep "$$delay"; \
	done; \
	printf "\n"; \
};

.PHONY: all banner install clean fclean re ff norm test help

# ========================================================
# Rules
# ========================================================
all: banner install $(NAME)
	@printf "$(RESET_LINE)$(GREEN)✅ $(NAME) is ready. Run: ./$(NAME)$(NOCOLOR)\n"

banner:
	@echo "🔨 Preparing environment..."

install:
	@$(DRAW_BAR) \
	$(PYTHON) -c "import pygame, flake8" 2>/dev/null || $(PIP) install -r requirements.txt --quiet --timeout 5 2>/dev/null || true; \
	draw_bar "Loading dependencies" "$(BLUE)" "📦" "0.07"

$(NAME):
	@$(DRAW_BAR) \
	printf '#!/bin/bash\nBASE_DIR="$$(cd "$$(dirname "$$0")" && pwd)"\ncd "$$BASE_DIR" || exit 1\nexport PYTHONPATH="$$BASE_DIR/$(SRCDIR):$$PYTHONPATH"\nexec $(PYTHON) "$$BASE_DIR/$(MAIN)" "$$@"\n' > $(NAME); \
	chmod +x $(NAME); \
	mkdir -p models; \
	draw_bar "Generating ./$(NAME)" "$(CYAN)" "🚀" "0.07"; \
	printf "\n$(GREEN)✅ Executable ready.$(NOCOLOR)\n"

clean:
	@$(DRAW_BAR) \
	find . -type f -name "*.py[co]" -delete; \
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -exec rm -rf {} + 2>/dev/null || true; \
	draw_bar "Cleaning pycache" "$(YELLOW)" "🗑️" "0.07"; \
	printf "\n$(GREEN)✅ Cache cleaned successfully.$(NOCOLOR)\n"

fclean: clean
	@$(DRAW_BAR) \
	rm -f $(NAME); \
	draw_bar "Removing ./$(NAME)" "$(RED)" "🗑️" "0.07"; \
	printf "\n$(GREEN)✅ Full clean complete.$(NOCOLOR)\n"

re: fclean all

ff norm:
	@$(PYTHON) -m flake8 $(SRCDIR)

test:
	@printf "$(CYAN)→ Running unit tests in $(TESTDIR)...$(NOCOLOR)\n"
	@$(PYTHON) -m unittest discover -s $(TESTDIR) -p "test_*.py" -v

# ========================================================
# Help & Documentation
# ========================================================
help:
	@echo "=========================================================="
	@echo "                 Learn2Slither Makefile                   "
	@echo "=========================================================="
	@echo "Available Make Targets:"
	@echo "  make          - Build executable ./$(NAME)"
	@echo "  make install  - Verify & install dependencies"
	@echo "  make clean    - Remove bytecode and __pycache__"
	@echo "  make fclean   - Full cleanup including ./$(NAME)"
	@echo "  make re       - Rebuild from scratch"
	@echo "  make norm     - Run flake8 norm check"
	@echo "  make test     - Run unit test suites"
	@echo ""
	@echo "Command Examples:"
	@echo "  ./snake --sessions 10 --save models/10sess.txt"
	@echo "  ./snake --visual --load models/100sess.txt --sessions 10 --dontlearn --step-by-step"
	@echo "  ./snake -vxl models/10ksess.txt -n 5 -step"
	@echo "=========================================================="