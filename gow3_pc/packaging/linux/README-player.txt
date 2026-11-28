The Last of Us Remastered - PC port (CUSA00552, update 1.11) - Linux

PORTUGUÊS

Este pacote NÃO traz nenhum arquivo do jogo. Você precisa da sua própria cópia descriptografada.

Requisitos: Linux de 64 bits (SteamOS do Steam Deck incluído) e placa de vídeo com driver
Vulkan 1.3 (Mesa ou NVIDIA). Não precisa instalar nada: FFmpeg, SDL3, o carregador Vulkan, X11
e as outras bibliotecas do jogo vêm no pacote (gow3_pc/out/lib). Do sistema ele usa só a glibc
(2.41 ou mais nova) e o driver de vídeo. licenses/THIRD-PARTY.txt lista tudo.
Não precisa instalar Python.

1. Extraia o zip em uma pasta com bastante espaço livre.
2. Abra "TLOU-Launcher" (no terminal: ./TLOU-Launcher).
3. Clique em "Pastas" e escolha "Arquivos do jogo": a pasta que tem o eboot.bin.
   A atualização 1.11 é procurada ao lado dela, na pasta de mesmo nome com "-patch" no fim
   (por exemplo, CUSA00552 e CUSA00552-patch).
4. Clique em JOGAR. A primeira vez demora mais, porque os shaders são compilados; podem
   aparecer travadas até eles entrarem no cache.

"TLOU-Game" inicia o jogo direto com as últimas configurações salvas no launcher.
Saves, configurações e logs ficam em analysis/ dentro desta pasta. Ao abrir, o launcher procura
uma versão nova do port; se houver, o botão ATUALIZAR baixa e instala sem mexer em analysis/.
Ao atualizar à mão, guarde essa pasta. O log é analysis/tlou/boot.log (botão "Abrir log").

ENGLISH

This package has NO game files. You need your own decrypted copy of the game.

Requirements: 64-bit Linux (SteamOS on the Steam Deck included) and a graphics card with a
Vulkan 1.3 driver (Mesa or NVIDIA). Nothing to install: FFmpeg, SDL3, the Vulkan loader, X11 and
the game's other libraries come with the package (gow3_pc/out/lib). From the system it uses only
glibc (2.41 or newer) and the graphics driver. licenses/THIRD-PARTY.txt lists them all.
No Python install is needed.

1. Extract the zip to a folder with plenty of free space.
2. Open "TLOU-Launcher" (from a terminal: ./TLOU-Launcher).
3. Click "Folders" and choose "Game files": the folder with eboot.bin. Update 1.11 is looked for
   beside it, in the folder of the same name ending in "-patch" (CUSA00552 and CUSA00552-patch).
4. Click PLAY. The first start takes longer while shaders compile, and you may see stutters until
   they are in the cache.

"TLOU-Game" starts the game directly with the settings last saved in the launcher.
Saves, settings and logs are in analysis/ inside this folder. When it opens, the launcher looks
for a new version of the port; if there is one, the UPDATE button downloads and installs it
without touching analysis/. Keep that folder when you update by hand.
The log is analysis/tlou/boot.log ("Open log").

Not affiliated with Sony, Naughty Dog or any other rights holder.
GNU GPL v2 or later (LICENSE.txt); the licenses folder lists the bundled libraries.
