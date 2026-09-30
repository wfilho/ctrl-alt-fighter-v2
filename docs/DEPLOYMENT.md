# Deploy do Site v2

## Publicacao Windows

- O fluxo de preparacao precisa iniciar o pnpm pelo Node diretamente quando o
  ambiente nao consegue iniciar `pnpm.cmd` via `spawn`.
- O empacotador usa Bash do Git. No Windows, informe o arquivo de archive como
  caminho MSYS (`/c/Users/...`) para o `tar`; o caminho retornado pelo workflow
  pode ser convertido para o caminho Windows antes de chamar o conector Sites.
- O Site publico deste fork e `ctrl-alt-fighter-v2-bbs-one`.
