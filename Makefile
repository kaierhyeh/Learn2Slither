SHELL   := /bin/bash
NAME     = snake
SRCDIR   = src
TESTDIR  = tests
MAIN     = $(SRCDIR)/main.py
PYTHON   = python3
PIP      = pip3

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
	draw_bar "Loading dependencies" "$(BLUE)" "📦" 0.07

$(NAME):
	@$(DRAW_BAR) \
	printf '#!/bin/bash\nBASE_DIR="$$(cd "$$(dirname "$$0")" && pwd)"\nexport PYTHONPATH="$$BASE_DIR/$(SRCDIR):$$PYTHONPATH"\nexec $(PYTHON) "$$BASE_DIR/$(MAIN)" "$$@"\n' > $(NAME); \
	chmod +x $(NAME); \
	mkdir -p models; \
	draw_bar "Generating ./$(NAME)" "$(CYAN)" "🚀" 0.07 \
	printf "\n$(GREEN)✅ Executable ready.$(NOCOLOR)\n"

clean:
	@$(DRAW_BAR) \
	find . -type f -name "*.py[co]" -delete; \
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -exec rm -rf {} + 2>/dev/null || true; \
	draw_bar "Cleaning pycache" "$(YELLOW)" "🗑️" 0.07; \
	printf "\n$(GREEN)✅ Cache cleaned successfully.$(NOCOLOR)\n"

fclean: clean
		@$(DRAW_BAR) \
	rm -f $(NAME); \
	draw_bar "Removing ./$(NAME)" "$(RED)" "🗑️" 0.07; \
	printf "\n$(GREEN)✅ Full clean complete.$(NOCOLOR)\n"

re: fclean all

ff norm:
	@$(PYTHON) -m flake8 --max-line-length=100 --ignore=E203,W503 $(SRCDIR)

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
	@echo "Subject Command Examples:"
	@echo "  ./snake -sessions 10 -save models/10sess.txt -visual off"
	@echo "  ./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step"
	@echo "  ./snake -visual on -load models/1000sess.txt"
	@echo "=========================================================="

# # Real dynamic bar
# PACKAGES = pygame flake8
# # ========================================================
# # Draw progress bar [████░░░░]
# # ========================================================
# DRAW_BAR = draw_bar() { \
# 	local cur=$$1 total=$$2 msg=$$3 color=$$4 icon=$$5 extra=$$6; \
# 	local pct=$$(( cur * 100 / total )); \
# 	local filled=$$(( cur * 20 / total )); \
# 	local empty=$$(( 20 - filled )); \
# 	local full="████████████████████"; \
# 	local blank="░░░░░░░░░░░░░░░░░░░░"; \
# 	local bar="$${full:0:filled}"; \
# 	local space="$${blank:0:empty}"; \
# 	printf "$(RESET_LINE)$${color}→ $${icon}  %-24s [%s%s] %3d%% $${extra}$(NOCOLOR)" "$$msg" "$$bar" "$$space" "$$pct"; \
# };

# install:
# 	@$(DRAW_BAR) \
# 	pkgs="$(PACKAGES)"; \
# 	total=$$(echo $$pkgs | wc -w); \
# 	count=0; \
# 	for pkg in $$pkgs; do \
# 		$(PYTHON) -c "import $$pkg" 2>/dev/null || $(PIP) install "$$pkg" --quiet --timeout 5 2>/dev/null || true; \
# 		count=$$((count + 1)); \
# 		draw_bar "$$count" "$$total" "Loading dependencies" "$(BLUE)" "📦" "($$pkg)"; \
# 	done; \
# 	printf "\n"

# $(NAME):
# 	@$(DRAW_BAR) \
# 	printf '#!/bin/bash\n' > $(NAME); \
# 	draw_bar 1 4 "Generating ./$(NAME)" "$(CYAN)" "🚀" "(Script Header)"; \
# 	printf 'BASE_DIR="$$(cd "$$(dirname "$$0")" && pwd)"\n' >> $(NAME); \
# 	printf 'export PYTHONPATH="$$BASE_DIR/$(SRCDIR):$$PYTHONPATH"\n' >> $(NAME); \
# 	draw_bar 2 4 "Generating ./$(NAME)" "$(CYAN)" "🚀" "(PYTHONPATH)"; \
# 	printf 'exec $(PYTHON) "$$BASE_DIR/$(MAIN)" "$$@"\n' >> $(NAME); \
# 	draw_bar 3 4 "Generating ./$(NAME)" "$(CYAN)" "🚀" "(Exec Entry)"; \
# 	chmod +x $(NAME); \
# 	mkdir -p models; \
# 	draw_bar 4 4 "Generating ./$(NAME)" "$(CYAN)" "🚀" "(Permissions)"; \
# 	printf "\n$(GREEN)✅ Executable ready.$(NOCOLOR)\n"

# clean:
# 	@$(DRAW_BAR) \
# 	items=$$(find . -type f -name "*.py[co]" -o -type d -name "__pycache__" -o -type d -name ".pytest_cache" 2>/dev/null); \
# 	total=$$(echo "$$items" | grep -v '^$$' | wc -l); \
# 	if [ "$$total" -eq 0 ]; then \
# 		draw_bar 1 1 "Cleaning pycache" "$(YELLOW)" "🗑️" "(0 items)"; \
# 	else \
# 		count=0; \
# 		for item in $$items; do \
# 			rm -rf "$$item"; \
# 			count=$$((count + 1)); \
# 			draw_bar "$$count" "$$total" "Cleaning pycache" "$(YELLOW)" "🗑️" "($$count/$$total)"; \
# 		done; \
# 	fi; \
# 	printf "\n$(GREEN)✅ Cache cleaned successfully.$(NOCOLOR)\n"

# fclean: clean
# 	@$(DRAW_BAR) \
# 	if [ -f "$(NAME)" ]; then \
# 		draw_bar 1 2 "Full cleanup" "$(RED)" "🗑️" "(Removing ./$(NAME))"; \
# 		rm -f $(NAME); \
# 		draw_bar 2 2 "Full cleanup" "$(RED)" "🗑️" "(Completed)"; \
# 	else \
# 		draw_bar 1 1 "Full cleanup" "$(RED)" "✅" "(Already clean)"; \
# 	fi; \
# 	printf "\n$(GREEN)✅ Full clean complete.$(NOCOLOR)\n"