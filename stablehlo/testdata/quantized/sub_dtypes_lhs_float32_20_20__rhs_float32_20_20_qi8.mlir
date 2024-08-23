// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<20x20xf32> {
    %cst = stablehlo.constant dense<"0x2F672D4004B95C401AB196BFBFD9C83F939C953ED1A92CC0F511F5BF33D8B6BEAE6D52C02C4AE2BF86D7A8C0213F96C0729B5CC0757ADBBFA2C104C03C924CBFB937C84080C53E409541464068F396C02C0218C056C7B840D23B5440E84D1640DB42D7BE8FD613401D8162C06689A73F5521A9401046F8BF2C27F5BFE759A0BF5A46BDBF908FBA3E43776EBF26C21D4008ED63C0DC7D13C0131271BE0E180ABF51D8C23F6B1B723F1E5D67C0BCDA214070208BBF1AF632C0140790BF4574DC3D367F25402BD92FC003DA42C0CE29A2C0E5D74CBC4E70463E99D19F401F1528403FADFABF6A5E8540C397A6BE94198740CC9A094051283C40F8723740901991C08FB066405FB336C028BD26401A3BE63FF52E4B405A751740E07C7FBF737137C00AE4AD4014B4B6BE42D14D40BABFE3403C0F14408A7AB13F946DA44045E016C08DE6DEBDB60E35C05105C53F4241443FE50A9E3F643A943F607F103EA0EC4ABFB8BEAC403EB6A23FEAC4A6C01B206D3FDD8B06C0E2DB4CC0E6F5A4C094CBAEBE1042C1C0DF3F7CBF3AB3A73FD18DD13FF087F2BFA08C303F0D295BC0789F833EF6FA34C06E828FC0478022C0793810C0FF2FD83E901A1EC0B11823404FF713C0357F92C01D8B35C075D312C0D35A873D8E6FC840949E6E4010009AC00FE4B03E44EDB9BE7CE667BF1B2C3DBF882237BCE00CBA3F45D429BFEDDA3CC0B4F44B407E4F873E0C37F33F1B690C4002A9693F261FBA408DB734C066925CBF358B51BF3507EEBE0CA3943FD8CB62BFD4D7DF3E17D64DC0AC46E63FE7BCB7BD95F102C0A24383C0548204407F0ABAC05B6511C08B6FA73F634B814005EAAE4034B3ACC03880DDC0653E9040F7625A4021068840ED103EBF3F5D12C0C7B647C028D39CBE0BEAD33F1F5F174015DA303FFBB4C4C00CDE42400DE87BC082642DC01AB396409907F23ED43045BFBD54B13FF6449240512B7AC03C56CDC01A3E93C0959A913F803CE7BF663A02C0574EC3403F04603F3254544008D2093E5B8CA4C039A68E401AD997C0620FE6BFDF5A8BBF6D71F63F62BC18C020A87040386516BF8B7897C0CBFC28406E228BBD1C4082C0471E933F2EF72CC011F43240718AAF3C0D9461C0493E8A3FE1FC573C45AA473ED6C60F40DB80A23F29C016C0C20782BF0A9814BFC68A9EBF040974C0DB461ABD8922A9BF1774CE3E47C6FEBF44B9FABFA5FFAB3EDE78BBBF165E213F2AD9903E28DD994046B39F40A5B086BF324832C0A776B4BF844D03C0A40216C00AB784C041207F40EF4CC7BE1838B83E43B69DC064B4E33ECCE91BC071F81BC07C7419403CA6A940B58598C0A9B0273ED9C6A240808CB4BD3A4EB03F1306F9BF07FA69BD3A9FCCBF0C369EBE8480FD3FA136A9C0E9E1D7BF64DA573F966217C07DB7074064B132C07CB383BFE5135D3FED9733C0AFA80E3F7ED385C0E99491C09DA35D40BFA3EEBFCC24963F19C06EBFBAF6AA3F41E84D3FA9359BC0D14F1BBFB98947C07D5DDABFC9A91C40CBF46C3E9D25D63F2881A8BE0BEA4DBFE333F93F615D843F13A938C0569B7F3E12A34EBEC8D97FC0D19576403DDFF9BFCA5A8DBF641D0ABE9395FE3E9638CE3FEF0C3A3F8B1B88C06DF651BF10052C40F8F654C02983C3402F0BE3BD1C4E03C1D7998D40A51640401D3B99BFDD2618BFE7B17740E705A140273C1040952789BFF9E3863F8A49513D49DCC2407190D74003E1653F72DE56C05F87D7C0E4C22B40473BB13C63CA03404D2659C06E9B083F744B5EBF2C60933ECB429EC03EB387403B9ED4BFA4829FC005DAFEBFB6C3D9BFDBE8A73FD99BFABFBA841140D68A12C0184557C0E0BFAAC040BCC33FCF3C9BC064202FC0305016C056430F40F4205C3EF15C4A405D4881BEA1F625401E60403E5F5EB13F76B0933FE91A0C40C4AA023EA6021AC0F96F4DC0D9C9D43F84493A40311A63BE448B473F180E0B40DC67A2BFEE8726408DA458C0BCCBE3BF379186C0D162434065B88FC0FE39D43F5EAD433FFD8E863E3988B33FD73A0A40B142913F0DF6AE3F2F7A3D405570A7C003F0623FB9147AC02407FFBEE03343C02DEBEB3FCB8EE43FA4C628C054D00F3FDACE9440669131C0A0C68C4024ADEDBEDC4E853F9D75C3400CD00D40792655C08132803F87E44B3FF686803F5E0C224040D182BF044E94BF841537C02CC8F5C09D99C4BFB8F992C0CEDF9CBF3EB66C3FF5A53AC02D6694BD1AA989C0CE0873C06C57F2BF44640FC02E79A140BBE6863F"> : tensor<20x20xf32>
    %cst_0 = stablehlo.constant dense<"0x8BA18BC0D3E3A5400B368A3E054F5E40688BBEBF36CA27BC1355833F83D7E0BF049E63C013E668C006C6423EB4F966C0F6B2A6BF703F99BF5A5E8340BA556BBF66D745BF7B710CC019578E3E1D7A2DC01C648B40A323A93FAA262B3FAF92F13FF4ADCBC030F07340089AC7BF8B604AC075BFB93F8B43A04018888CBF092EF33E2CD0ECBD7A9D0AC07176B63FC2331F40AD8D06400CFA9CBF1EF7983F45A99BBD100ECEBDA24000C1C3D542C0A0FF5240737EA94036DBDD40C0375EC00C6DA2BECC09BB40257C63402EA90D40B7F228BE597DDFBFAD29A13E51E03DBEEB9386C0D8B055C0F15F9ABC0561A2C04572F33F1508CFC025CB9140F73EDFBF4080373D5D02EB3FC8D8ED3FD2B99CBD3673D63D7278863FA2DC613F82EC4D4032B427407972E33F726BD23D6E8B91BF1D2A9EBF728B3E40FA2CAD3FD9344CBF5D6A37C0CEF5E53E1C6A7840906678C091284C40A630ADBF246A82BEE1204DC08AF3FF3FAD021F3FB530ED403356EBBF3C91A2BDCAB40D419CD08B404C058540F7496D40D4BE50BFDAC7A0BFA294003FD03B183FD7BA5DC0F2E9A64045EFFB3F4E9934402F8B61BFD8FEEAC0847A81BFF53E5DBF54CE50C0553AC2C0916389BFE126D0BF5800673EB29F383F24FDBEC037B9F2BE6A9F353F6723A64043997D40D74D3EC0B33248C03FE77FC0EF51924030261C40212FFFBFFAAB02BE32AA933F461EFD3FC7049ABF00E948C028B6D8BEA15CE3BF74415D40DEC77BBEFEBAE2BF024C47C0DB03ABC0548C75C07F07D03F6DD837C08EFA8D40F42D54C0A3AB2A40A2D44A3C3CBB033FA7493140552C6CC0AB14F2C0647A87C0B451FA3F8BE2523E41B60FC0919835404CE4AEC03CF93F40EED380BF04B946C056FA743F180155C0523FC04030248ABF0E793740F4903340AC45B83FCB1518400BE000BE68F71D40A2DDC3BF0CF546400CF13CC0B07DB53F68A59BC0C097EC3E3B0DF8BFD7530AC053F16C3FE00C0940B0713DBE8CD404C0D3C94E402DB651BEE5F83B40A2C2093F8D3D79407F64B53FFBD233C08D365A3FD68B893F0B43D8BFDB0341C0EAC15F3D4D2D8ABEE5AE1F3F214BE43F7D958AC0BA0A69C01D376740205B943FCD496AC09E143BC0A869CA3F151804C0D67F2EBF6ED22CBE87046540222C8BC08EF8DF3F9813A53E23E4C040CE54ABC06093104050D2303F3AE939408A048FBF31AD60C099E619C06E70CDBDE53386BFBDC23AC0DC3B1BC05BBDAEC023910C40CC03CC407308E23F23CA32C0F79DF23E42B51340968D27C0FE6229C0FC8DA0BF4715A33D3B86063EB7A356C0FD3B8DC08B8894C0EC66813DD94687BF5002A94062F7F73EC03100BFF46A84BE14745F40EC388EC0D71002BE3E807F3F0E28664099A9F6BEB70719BC09842DC0D053E4BF1EC0F53FD7799C401C8472C022B6CBBEFE9149C040277240971882C070ADB7C0EC86993FDA2C35405CB5CC3F42C53740CFD8FCBFDE4B424026072DBF98878F408EEAB4BF4BE6913F141AADC08FDD9840C9E63740DA5F88C075AEBBBFDFB0B43F1711334008D74FC0DAC85240BA28AF3ED8B63BBF83BDB23F1C2A94BE47394FC0D991903F98841BBF78A27340C56D87C04A7B6C3F5419423F5F3486401FA7A1BF49B327C0E7B8193F8B2DEDBECAB1724051130EC0AF971A40DAD2CAC0C7CACA3ED0778FC077FBE43FB541B3BF2C79B740C0B73D4047D1FBBF3F2E953FE6DEB34022B49440B11BFD3F450673BF23D3D93F70618940F1B5EA3EA26F7F40150A36C070BD6CC0F7B3EEC00FE3993FA5E606BEA7C186C02452763FA05C75405BC682403CAF833FE184A9BE1681B1BFD73FA2BFEC2847BF0E644E40CDFABB3F81E939BF91F6023FA8A6B13FD90A3EBD5320F53F293367BFF76286BEAFC8FBBF81CFAD3FA62AC43CBF6A4DC0A6A830C0ECC6CB3F70F03EBF19800040D1DB52C04788BA3E4882C13FAD7EE5BE85E0E13F3C948DBF7A21EBBE7B44B2BFD9A11B3F0EEEC73F443C1FC06674DCBF8D7049BFAD0A6D3D4B617AC0B16703C104341AC0A94CBF4071BA3F4033C2C5BADCACC2BF3E7F2040D6017E3F7931BA3F07F994C06BD384C0F1758AC0B7E022BF505EDB40C196DBC04C5378C0D1D334C0886E8A3F9227943F34C7AD4058ECB5BFD61E0D3FA89B133F6E1296BF95F0ECBC48F1313F54F28940113ABCBFAA10D1C0B993A33F6BEA8CC003B15A3F9D646AC0F9E396C046AA333FFAC98F4090D4423FFAE48240636236C063C92B3F02027EC0"> : tensor<20x20xf32>
    %cst_1 = stablehlo.constant dense<"0x563C7F3F00000000FF948BBE00000000D38B973E00000000DD3D7DBF000000000000000000000000236747BE000000000000000000000000DD3D7DBF00000000563C7F3F563C7F3FD671393F00000000DD3D7DBF000000008A7CAB3E00000000000000000000000000000000563C7F3F00000000DD3D7DBF000000008245F3BE000000004F70BB3EDD3D7DBF00000000DD3D7DBF00000000DD3D7DBF00000000563C7F3F0947713F0000000000000000DD3D7DBFDD3D7DBF00000000CB54DF3D00000000DD3D7DBFDD3D7DBF0000000000000000563CFFBD563C7F3F563C7F3F00000000563C7F3F0000000000000000563C7F3F00000000563C7F3F406D3FBD00000000DD3D7DBF563C7F3F3550653F000000009148EF3DDD3D7DBFDD3D7DBF000000000661CFBD563C7F3F563C7F3F0000000000000000563C7F3F00000000AE4EE7BEDD3D7DBF563C7F3F91486FBE563C7F3F563C7F3FF0910F3EDD3D7DBF326AC33E0000000000000000184A6D3FDD3D7DBFDD3D7DBFDD3D7DBFDD3D7DBF0000000000000000563CFF3E0661CF3E00000000B6859FBEDD3D7DBFC86E3DBF00000000000000000000000000000000E95AD73E00000000563C7F3F00000000AE4E67BED67139BF00000000F0918F3DE28E933E00000000DD3D7DBF7B79AF3E0000000000000000DD3D7DBFDD3D7DBF563C7F3F00000000DD3D7DBF000000000E98873E563C7F3F563C7F3F274D693F0000000000000000000000000000000000000000563C7F3FDD3D7DBFCB54DF3EDD3D7DBF563C7F3FDD3D7DBF563C7FBC1C9B03BF000000000000000000000000563C7F3F0000000015644B3F00000000DD3D7DBF563C7F3F00000000563C7F3F00000000FB4375BF00000000DD3D7DBF563C7F3F00000000B6859FBEDD3D7DBF0000000000000000DD3D7DBF563C7F3F959905BF0000000000000000563C7F3F9F4BEBBE0000000000000000B6859F3DDD3D7DBF00000000563C7F3F563CFFBD563C7F3F53565DBF869609BF00000000DD3D7DBF00000000615959BF0000000000000000563C7F3FCB545FBD00000000406DBF3EDD3D7DBF00000000563C7F3FDD3D7DBF00000000406DBF3C0000000000000000563C7F3C2367473E563C7F3F0000000000000000DD3D7DBFA782A3BEDD3D7DBF00000000DD3D7DBFF47731BF4C8A19BF00000000000000008A7CAB3E000000002E84213FF0918F3E563C7F3F563C7F3FDD3D7DBFDD3D7DBFDD3D7DBF000000008245F3BEDD3D7DBF563C7F3F000000005E73B73EB6859FBDB6859F3E0000000000000000563C7F3F91486F3F00000000705C55BF1C9B033F00000000563C7F3FDD3D7DBF0000000000000000DD3D7DBF000000000000000000000000E95A573F0000000000000000DD3D7DBF0000000053565D3F00000000BD51E3BE000000000000000000000000DD3D7DBF00000000DD3D7DBF563C7F3F236747BE00000000DD3D7DBF00000000DD3D7DBF563C7F3FAB6845BF000000000000000000000000000000000000000000000000406D3FBF7B79AFBE00000000000000000000000000000000DD3D7DBF563CFF3E00000000D671393F184A6DBFB96B41BF0000000000000000563C7F3F4C8A19BF0000000000000000563C7F3FDD3D7DBF00000000C4881B3F563C7F3F000000000000000000000000824573BF563C7F3F000000000661CFBDDD3D7DBFDD3D7DBF563C7F3FEC4079BF000000009F4BEBBE9148EFBE00000000E28E933E00000000000000000000000000000000FB4375BFDD3D7DBF00000000DD3D7DBF563C7F3F00000000000000000000000000000000DD3D7DBF00000000A49C01BF00000000CB545F3E0000000000000000563C7F3F406D3F3E00000000EC40793F563C7F3F0E98073EDD3D7DBF0000000000000000563C7F3F4F70BBBECB545FBE563C7F3FDD3D7DBF563C7F3F0000000000000000C4881BBF0000000000000000563C7F3F326A433F06614F3E563C7F3F563C7F3F563C7F3F000000000000000000000000BD51633FDD3D7DBFDD3D7DBFDD3D7DBF563C7F3F563C7F3F00000000F0910F3F0000000000000000563C7F3F0000000000000000000000000000000000000000BD51E33ECB545F3E563C7F3F563C7F3FF47731BFDD3D7DBF0000000000000000DD3D7DBF00000000DA575BBF184A6D3F000000006C7633BFDD3D7DBF326A43BFDD3D7DBF00000000987FA73E563C7F3F"> : tensor<20x20xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<20x20xf32>) -> tensor<20x20x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<20x20xf32>) -> tensor<20x20x!quant.uniform<i8:f32, 0.0039213846711551445:-128>>
    %2 = stablehlo.subtract %1, %0 : (tensor<20x20x!quant.uniform<i8:f32, 0.0039213846711551445:-128>>, tensor<20x20x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<20x20x!quant.uniform<i8:f32, 0.0077891749494216024:-1>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<20x20x!quant.uniform<i8:f32, 0.0077891749494216024:-1>>) -> tensor<20x20xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_1, %3) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<20x20xf32>
  }
}
